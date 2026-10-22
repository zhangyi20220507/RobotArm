"""No CAN, cameras or GPU: delayed policy and clock alignment contracts."""
import tempfile
import time
import unittest
import json
import runpy
from pathlib import Path

import torch

from robotarm.modeling.online_policy import (
    AsyncPlanner, parse_args, pi0_expired_steps, trim_pi0_plan, pi0_checkpoint_digest, validate_approval,
)


class DelayedPolicy:
    def reset(self):
        pass

    def select_action(self, batch):
        raise AssertionError('Pi0 must generate one full chunk, not select in a loop')

    def predict_action_chunk(self, batch):
        time.sleep(.4)
        return torch.arange(50.).reshape(1,50,1).expand(1,50,7)


class Pi0AsyncTests(unittest.TestCase):
    def test_400ms_inference_does_not_block_control_and_old_prefix_is_skipped(self):
        planner=AsyncPlanner(DelayedPolicy(),lambda x:x,lambda x:x,None,20,full_chunk=True)
        planner.start()
        try:
            started=time.monotonic()
            request=planner.request({},started)
            ticks=0
            while time.monotonic()-started<.35:
                self.assertIsNone(planner.take_ready(request))
                ticks+=1
                time.sleep(.01)
            self.assertGreaterEqual(ticks,15)
            plan=planner.wait_ready(2,request)
            self.assertIsNotNone(plan)
            self.assertEqual(len(plan['actions']),50)
            actions,skip=trim_pi0_plan(plan['actions'],plan,started+.4,20,20)
            self.assertEqual(skip,8)
            self.assertEqual(len(actions),12)
            self.assertEqual(actions[0][0],8.)
        finally:planner.close()

    def test_expired_plan_cannot_replay_and_request_generation_filters_old_trial(self):
        actions=list(range(50))
        kept,skip=trim_pi0_plan(actions,{'requested_at':0},1.01,20,20)
        self.assertFalse(kept)
        self.assertGreater(skip,20)
        planner=AsyncPlanner(None,None,None,None,20)
        planner._results.put({'request_id':1})
        self.assertIsNone(planner.take_ready(minimum_request_id=2))

    def test_clock_is_stable_at_exact_boundaries(self):
        self.assertEqual(pi0_expired_steps(0,.4,20),8)
        self.assertEqual(pi0_expired_steps(0,.4001,20),9)
        self.assertEqual(pi0_expired_steps(1,0,20),0)

    def test_pi0_digest_changes_with_base_and_processors(self):
        with tempfile.TemporaryDirectory() as root:
            cp,base=Path(root)/'cp',Path(root)/'base'
            cp.mkdir();base.mkdir()
            for p in [cp/'adapter_model.safetensors',cp/'policy_preprocessor.json',
                      base/'config.json',base/'model.safetensors']:p.write_bytes(b'a')
            original=pi0_checkpoint_digest(cp,base)
            (base/'model.safetensors').write_bytes(b'b')
            self.assertNotEqual(original,pi0_checkpoint_digest(cp,base))
            changed=pi0_checkpoint_digest(cp,base)
            (cp/'policy_preprocessor.json').write_bytes(b'c')
            self.assertNotEqual(changed,pi0_checkpoint_digest(cp,base))

    def test_generic_aliases_parse_and_execution_is_opt_in(self):
        args=parse_args(['--checkpoint','unused','--max-plan-age-ms','1000',
                         '--replan-lead-steps','10'])
        self.assertEqual(args.max_smolvla_plan_age_ms,1000)
        self.assertEqual(args.smolvla_replan_lead_steps,10)
        self.assertFalse(args.execute)
        self.assertFalse(args.experimental_pi0_execute)

    def test_pi0_approval_binds_alignment_inference_and_raw_limits(self):
        from robotarm.modeling.approve import build_approval
        row_factory=runpy.run_path(str(Path(__file__).with_name('test_approve_deadline.py')))['_row']
        row=row_factory(policy_type='pi0',task='Pick up the cup',chunk_size=50,n_action_steps=20,
            planner_mode='async_chunked',planner_underflow=False,chunk_refresh=True,
            chunk_inference_ms=400.,plan_observation_age_ms=450.,max_chunk_inference_ms=600.,
            smolvla_replan_lead_steps=12,pi0_clock_aligned=True,num_inference_steps=10,
            raw_action=[0.]*7,safe_action=[0.]*7)
        with tempfile.TemporaryDirectory() as root:
            report,approval_path=Path(root)/'shadow.jsonl',Path(root)/'approval.json'
            report.write_text(json.dumps(row)+'\n')
            result=build_approval(report,approval_path,1,.05)
            self.assertTrue(result['act_raw_action_watchdog_passed'])
            self.assertEqual(result['runtime']['num_inference_steps'],10)
            self.assertIn('max_act_raw_joint_gap_deg',result['runtime'])
            approval_path.write_text(json.dumps(result))
            validate_approval(approval_path,'0'*64,'pi0',result['runtime'])
            changed=dict(result['runtime'],num_inference_steps=5)
            with self.assertRaises(RuntimeError):
                validate_approval(approval_path,'0'*64,'pi0',changed)
            row['pi0_clock_aligned']=False
            report.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(SystemExit,'clock alignment'):
                build_approval(report,approval_path,1,.05)


if __name__=='__main__':unittest.main()
