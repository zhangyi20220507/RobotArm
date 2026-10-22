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

    def run_js(self, assertions, stored="zh-CN", storage_blocked=False):
        if not shutil.which("node"):
            self.skipTest("node not installed")
        harness = r'''
const vm=require('node:vm'), assert=require('node:assert/strict');
class Element {
  constructor(tag='div',attrs={}){
    this.tag=tag;this.attrs=attrs;this.dataset={};this.children=[];
    this.value=attrs.value||'';this.textContent='';this.innerHTML='';this.disabled=false;
    this.classList={toggle(){}};
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
current={index:0,length:400,duration:20,metrics:{score:80,max_joint_step_deg:6,p95_tracking_deg:1,
max_gripper_step_mm:2,p95_gripper_tracking_mm:3,timestamp_jitter_ms:1,idle_ratio:.1,warnings:['动作存在大于 5°/帧的关节跳变']}};
$('#trimStart').value='5';$('#trimEnd').value='14';$('#note').value='不要翻译我的备注';$('#task').textContent='抓杯子';
time=7;dirty=true;playing=true;selectedStatus='keep';message('#saved','未保存');`);
el('#languageToggle').onclick();
assert.equal(storage,'en');assert.equal(document.documentElement.lang,'en');
assert.equal(el('#languageToggle').textContent,'中文');
assert.equal(el('#search').attrs.placeholder,'Search ID / task');
assert.equal(el('#prev').attrs.title,'Shortcut: ↑');
assert.equal(nodes.find(x=>x.tag==='option'&&x.attrs.value==='keep').textContent,'Keep');
assert.equal(el('#play').textContent,'❚❚ Pause');
assert.equal(el('#saved').textContent,'Unsaved changes');
assert.match(el('#trimSummary').textContent,/180 frames/);
assert.match(el('#warnings').textContent,/Joint action jumps/);
assert.match(el('#metrics').innerHTML,/Full-episode quality score/);
assert.match(el('#dataset').textContent,/我的数据集/);
assert.equal(el('#note').value,'不要翻译我的备注');assert.equal(el('#task').textContent,'抓杯子');
assert.equal(el('#trimStart').value,'5');assert.equal(el('#trimEnd').value,'14');
assert.equal(run('time'),7);assert.equal(run('dirty'),true);assert.equal(run('playing'),true);
assert.equal(run('selectedStatus'),'keep');assert.equal(calls,1);
run(`message('#exportStatus',()=>t('exportProgress',{phase:t('编码视频'),done:20,total:180,output:'/tmp/中文路径'}));`);
assert.match(el('#exportStatus').textContent,/Encoding videos/);
el('#languageToggle').onclick();
assert.equal(storage,'zh-CN');assert.match(el('#exportStatus').textContent,/编码视频/);
assert.match(el('#exportStatus').textContent,/中文路径/);assert.equal(el('#saved').textContent,'未保存');
assert.equal(calls,1);
''')

    def test_remembers_english_on_page_load(self):
        self.run_js("assert.equal(document.documentElement.lang,'en');assert.equal(el('#play').textContent,'▶ Play');", stored="en")

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
        self.run_js("assert.equal(document.documentElement.lang,'zh-CN');el('#languageToggle').onclick();assert.equal(document.documentElement.lang,'en');", storage_blocked=True)

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
