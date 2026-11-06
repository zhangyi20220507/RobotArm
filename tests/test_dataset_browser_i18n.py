"""Exercise language switching without a browser, network, dataset, or hardware."""
import json
import re
import shutil
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path


HTML_PATH = Path(__file__).resolve().parents[1] / "src/robotarm/modeling/dataset_browser.html"


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.nodes = []

    def handle_starttag(self, tag, attrs):
        self.nodes.append({"tag": tag, "attrs": dict(attrs)})


class DatasetBrowserLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = HTML_PATH.read_text()
        cls.script = cls.html.split("<script>", 1)[1].split("</script>", 1)[0]
        parser = Elements()
        parser.feed(cls.html.split("<script>", 1)[0])
        cls.nodes = parser.nodes
        cls.translations = json.loads(re.search(
            r"const translations = (\{.*?\});\n", cls.script, re.S).group(1))

    def test_all_marked_labels_and_attributes_have_both_languages(self):
        self.assertIn('id="languageZh" type="button" data-lang="zh-CN" aria-pressed="true">中文</button>', self.html)
        self.assertIn('id="languageEn" type="button" data-lang="en" aria-pressed="false">English</button>', self.html)
        for node in self.nodes:
            for attr, key in node["attrs"].items():
                if attr.startswith("data-i18n"):
                    self.assertIn(key, self.translations)
                    self.assertEqual(len(self.translations[key]), 2)
                    self.assertTrue(all(self.translations[key]))
        self.assertNotRegex(self.html, r"<option[^>]*>\s*<span")
        for zh, en in self.translations.values():
            self.assertEqual(set(re.findall(r"\{(\w+)\}", zh)),
                             set(re.findall(r"\{(\w+)\}", en)))

    def test_english_episode_summary_keeps_trim_on_second_line(self):
        self.run_js(r'''
run(`meta={name:'dataset',total_episodes:1,fps:20,
  reviews:{'0':{status:'keep',trim:{start_frame:41,end_frame:312}}},
  episodes:[{index:0,length:312,duration:15.6,tasks:[]}]};
  setLanguage('en');`);
const left=el('#episodes').children[0].children[0];
assert.equal(left.innerHTML,'Episode 0<span class="ep-duration"> · 15.6s</span><br><small>312f · trim 2.05–15.60s</small>');
run(`setLanguage('zh-CN')`);
assert.match(el('#episodes').children[0].children[0].innerHTML,/312 帧 · 15.6s · 已裁剪 2.05–15.60s/);
''')

    def test_episode_labels_render_filter_and_retranslate(self):
        self.run_js(r'''
run(`meta={name:'dataset',total_episodes:3,fps:20,label_schema:[{
  id:'cup_orientation',labels:{'zh-CN':'杯子姿态',en:'Cup orientation'},options:[
    {value:'upright',labels:{'zh-CN':'杯子直立',en:'Cup upright'}},
    {value:'sideways',labels:{'zh-CN':'杯子侧倒',en:'Cup sideways'}},
    {value:'uncertain',labels:{'zh-CN':'其他 / 不确定',en:'Other / uncertain'}}]}],
  reviews:{'0':{labels:{cup_orientation:'upright'}},'1':{labels:{cup_orientation:'sideways'}}},
  episodes:[0,1,2].map(index=>({index,length:100,duration:5,tasks:[]}))};
  selectedLabels={cup_orientation:'sideways'};renderLabelControls();renderList();`);
assert.equal(el('#labelFilter').children.length,5);
assert.equal(el('#exportLabelFilter').children.length,5);
assert.equal(el('#labelEditor').children[0].children[1].value,'sideways');
el('#labelFilter').value='cup_orientation=sideways';run(`renderList()`);
assert.equal(el('#episodes').children.length,1);
assert.match(el('#episodes').children[0].children[0].innerHTML,/杯子侧倒/);
run(`setLanguage('en')`);
assert.equal(el('#labelFilter').value,'cup_orientation=sideways');
assert.equal(el('#episodes').children.length,1);
assert.match(el('#episodes').children[0].children[0].innerHTML,/Cup sideways/);
assert.equal(el('#labelEditor').children[0].children[0].textContent,'Cup orientation');
''')

    def run_js(self, assertions, stored="zh-CN", storage_blocked=False):
        if not shutil.which("node"):
            self.skipTest("node not installed")
        harness = r'''
const vm=require('node:vm'), assert=require('node:assert/strict');
class Element {
  constructor(tag='div',attrs={}){
    this.tag=tag;this.attrs=attrs;this.dataset={};this.children=[];
    this.value=attrs.value||'';this.textContent='';this.innerHTML='';this.disabled=false;
    const classes=new Set();this.classList={toggle(name,on){if(on)classes.add(name);else classes.delete(name)},contains(name){return classes.has(name)}};
    for(const [key,value] of Object.entries(attrs))if(key.startsWith('data-'))
      this.dataset[key.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=value;
  }
  setAttribute(k,v){this.attrs[k]=v} getAttribute(k){return this.attrs[k]}
  append(...nodes){this.children.push(...nodes)} appendChild(node){this.append(node)}
  replaceChildren(...nodes){this.children=nodes} querySelector(){return null}
}
const nodes=NODES.map(x=>new Element(x.tag,x.attrs)), elements=new Map();
for(const node of nodes)if(node.attrs.id)elements.set('#'+node.attrs.id,node);
const document={documentElement:{},title:'',
  querySelector:id=>elements.get(id),
  querySelectorAll:selector=>{
    const match=selector.match(/^\[([^\]]+)\]$/);
    if(match)return nodes.filter(x=>match[1] in x.attrs);
    if(selector==='.review-buttons button')return nodes.filter(x=>'data-status' in x.attrs);
    if(selector==='.language-switch button')return nodes.filter(x=>'data-lang' in x.attrs);
    return [];
  },createElement:tag=>new Element(tag),addEventListener(){}};
let calls=0, storage=STORED;
const context={document,window:{addEventListener(){},localStorage:{
  getItem(){if(BLOCKED)throw Error('storage blocked');return storage},
  setItem(k,v){if(BLOCKED)throw Error('storage blocked');assert.equal(k,'robotarm.datasetBrowser.language');storage=v}
}},fetch(){calls++;return new Promise(()=>{})},requestAnimationFrame(){},setTimeout,clearTimeout,confirm:()=>true};
vm.createContext(context);vm.runInContext(SCRIPT,context);
const run=code=>vm.runInContext(code,context), el=id=>elements.get(id);
ASSERTIONS
'''
        harness = harness.replace("NODES", json.dumps(self.nodes)).replace("STORED", json.dumps(stored))
        harness = harness.replace("BLOCKED", str(storage_blocked).lower())
        harness = harness.replace("SCRIPT", json.dumps(self.script)).replace("ASSERTIONS", assertions)
        result = subprocess.run(["node"], input=harness, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_switch_updates_labels_messages_and_preserves_unsaved_state(self):
        self.run_js(r'''
assert.equal(document.documentElement.lang,'zh-CN');
run(`meta={name:'我的数据集',total_episodes:1,fps:20,reviews:{},episodes:[{index:0,length:400,duration:20,tasks:['抓杯子']}]};
current={index:0,length:400,duration:20,metrics:{max_joint_step_deg:6,p95_tracking_deg:1,
max_gripper_step_mm:2,p95_gripper_tracking_mm:3,warnings:['动作存在大于 5°/帧的关节跳变']}};
$('#trimStart').value='5';$('#trimEnd').value='14';$('#note').value='不要翻译我的备注';
time=7;dirty=true;playing=true;selectedStatus='keep';message('#saved','未保存');`);
el('#languageEn').onclick();
assert.equal(storage,'en');assert.equal(document.documentElement.lang,'en');
assert.equal(el('#languageZh').attrs['aria-pressed'],'false');
assert.equal(el('#languageEn').attrs['aria-pressed'],'true');
assert.equal(el('#languageEn').classList.contains('selected'),true);
assert.equal(el('#languageZh').classList.contains('selected'),false);
assert.equal(el('#search'),undefined);
assert.equal(nodes.find(x=>x.attrs.id==='statusFilter').tag,'select');
assert.equal(el('#prev'),undefined);
assert.equal(el('#next'),undefined);
assert.equal(nodes.find(x=>x.tag==='option'&&x.attrs.value==='keep').textContent,'Keep');
assert.equal(el('#play').textContent,'❚❚ Pause');
assert.equal(el('#saved').textContent,'Unsaved changes');
assert.equal(el('#trimSummary').textContent,'');
assert.equal(el('#trimMarkers').hidden,false);
assert.equal(el('#trimStartMarker').attrs.style,'left:25.0000%');
assert.equal(el('#trimEndMarker').attrs.style,'left:70.0000%');
assert.equal(el('#trimRange').attrs.style,'left:25.0000%;width:45.0000%');
assert.equal(el('#seek').attrs.title,'Trim start 5.00 s, end 14.00 s');
assert.match(el('#warnings').textContent,/Joint step > 5°\/frame/);
assert.equal(el('#warnings').className,'warning');
assert.equal(el('#qualityScore'),undefined);
assert.equal((el('#metrics').innerHTML.match(/class="metric"/g)||[]).length,4);
assert.match(el('#metrics').innerHTML,/Max joint step/);
assert.match(el('#metrics').innerHTML,/P95 gripper error/);
assert.equal(nodes.find(x=>x.attrs['data-i18n']==='当前帧设起点').textContent,'Set start');
assert.equal(nodes.find(x=>x.attrs['data-i18n']==='当前帧后设终点').textContent,'Set end');
assert.match(el('#dataset').textContent,/我的数据集/);
assert.equal(el('#note').value,'不要翻译我的备注');
assert.equal(el('#trimStart').value,'5');assert.equal(el('#trimEnd').value,'14');
assert.equal(run('time'),7);assert.equal(run('dirty'),true);assert.equal(run('playing'),true);
assert.equal(run('selectedStatus'),'keep');assert.equal(calls,1);
run(`message('#exportStatus',()=>t('exportProgress',{phase:t('编码视频'),done:20,total:180,output:'/tmp/中文路径'}));`);
assert.match(el('#exportStatus').textContent,/Encoding videos/);
el('#languageZh').onclick();
assert.equal(storage,'zh-CN');assert.match(el('#exportStatus').textContent,/编码视频/);
assert.equal(el('#languageZh').classList.contains('selected'),true);
assert.equal(el('#languageEn').attrs['aria-pressed'],'false');
assert.match(el('#exportStatus').textContent,/中文路径/);assert.equal(el('#saved').textContent,'未保存');
assert.equal(calls,1);
''')

    def test_remembers_english_on_page_load(self):
        self.run_js("assert.equal(document.documentElement.lang,'en');assert.equal(el('#play').textContent,'▶ Play');assert.equal(el('#languageEn').classList.contains('selected'),true);", stored="en")

    def test_replay_starts_at_full_or_selected_segment_start(self):
        self.run_js(r'''
run(`meta={fps:20};current={index:0,length:400,duration:20};time=12;playing=true;
$('#trimStart').value='5';$('#trimEnd').value='15';$('#previewTrim').checked=true;`);
el('#replay').onclick();
assert.equal(run('time'),5);
assert.equal(el('#seek').value,5);
assert.equal(run('playing'),true);
assert.equal(el('#play').textContent,'❚❚ 暂停');
run(`$('#previewTrim').checked=false;time=18;`);
el('#replay').onclick();
assert.equal(run('time'),0);
assert.equal(el('#seek').value,0);
run(`current=null;meta=null;setLanguage('en')`);
assert.equal(el('#replay').attrs.title,'Replay from start');
assert.equal(nodes.find(x=>x.attrs['data-i18n']==='↺ 重播').textContent,'↺ Replay');
''')

    def test_trim_markers_follow_timestamp_edits_and_hide_for_invalid_range(self):
        self.run_js(r'''
run(`meta={fps:20};current={index:0,length:400,duration:20};
$('#seek').max=19.95;$('#trimStart').value='2';$('#trimEnd').value='16';trimSummary();`);
assert.equal(el('#trimMarkers').hidden,false);
assert.equal(el('#trimStartMarker').attrs.style,'left:10.0251%');
assert.equal(el('#trimEndMarker').attrs.style,'left:80.2005%');
run(`$('#trimStart').value='18';trimSummary();`);
assert.equal(el('#trimMarkers').hidden,true);
assert.match(el('#trimSummary').textContent,/起止时间/);
run(`$('#trimEnd').value='20';trimSummary();`);
assert.equal(el('#trimMarkers').hidden,false);
assert.equal(el('#trimSummary').textContent,'');
''')

    def test_export_progress_lifecycle_and_language_switch(self):
        self.run_js(r'''
assert.equal(el('#exportProgress').hidden,true);
run(`updateExportJob({status:'running',total_episodes:30,episodes_done:0,current_episode:0,
episode_position:1,frames_done:0,total_frames:6000,phase:'复制帧',output:'/tmp/export'});`);
assert.equal(el('#exportProgress').hidden,false);
assert.equal(el('#exportProgressBar').max,30);assert.equal(el('#exportProgressBar').value,0);
assert.match(el('#exportCurrent').textContent,/原 Episode 0/);
run(`updateExportJob({...exportJob,episodes_done:12,current_episode:18,episode_position:13,
frames_done:2600,phase:'编码视频'});`);
assert.equal(el('#exportProgressBar').value,12);
assert.match(el('#exportEpisodeCount').textContent,/总计 30 条 · 已完成 12 条 · 40%/);
run(`setLanguage('en')`);
assert.match(el('#exportEpisodeCount').textContent,/Total: 30 episodes · Completed: 12 · 40%/);
assert.match(el('#exportCurrent').textContent,/Processing 13\/30 \(source Episode 18\) · Encoding videos/);
assert.match(el('#exportStatus').textContent,/2600\/6000 frames/);
assert.equal(el('#exportProgressBar').attrs['aria-valuetext'],el('#exportEpisodeCount').textContent);
run(`updateExportJob({...exportJob,status:'failed',error:'disk full'});`);
assert.equal(el('#exportProgressBar').value,12);assert.equal(el('#export').disabled,false);
assert.match(el('#exportStatus').textContent,/Export failed/);
run(`updateExportJob({...exportJob,status:'running',episodes_done:0,current_episode:null});`);
assert.equal(el('#exportProgressBar').value,0);
run(`updateExportJob({...exportJob,episodes_done:30,frames_done:6000,phase:'生成统计与来源清单'});`);
assert.equal(el('#exportProgressBar').value,30);assert.equal(el('#export').disabled,true);
assert.doesNotMatch(el('#exportStatus').textContent,/Complete:/);
run(`updateExportJob({...exportJob,status:'complete',episodes:30,frames:6000});`);
assert.match(el('#exportStatus').textContent,/Complete: 30 episodes/);
assert.equal(el('#export').disabled,false);
run(`setLanguage('zh-CN')`);
assert.match(el('#exportStatus').textContent,/完成：30 条/);
assert.equal(el('#exportProgressBar').value,30);assert.equal(calls,1);
''')

    def test_storage_failure_falls_back_without_breaking_switch(self):
        self.run_js("assert.equal(document.documentElement.lang,'zh-CN');el('#languageEn').onclick();assert.equal(document.documentElement.lang,'en');", storage_blocked=True)

    def test_localized_range_and_server_errors_retranslate(self):
        self.run_js(r'''
run(`setLanguage('en');message('#saved',new LocalizedError('rangeError',{duration:20}),true);`);
assert.match(el('#saved').textContent,/start < end/);
run(`setLanguage('zh-CN')`);assert.match(el('#saved').textContent,/起点 < 终点/);
run(`setLanguage('en');message('#saved',new Error('输出目录已经存在，请换一个新名称'),true);`);
assert.match(el('#saved').textContent,/already exists/);
run(`setLanguage('zh-CN')`);assert.match(el('#saved').textContent,/已经存在/);
''')


if __name__ == "__main__":
    unittest.main()
