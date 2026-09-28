// Synthetic UI evidence tests only; no encoder, identity-quality or browser claim.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(__dirname+'/../explanation_ui.js','utf8');
class Element{
  constructor(){this.children=[];this.style={};this.hidden=false;this.textContent='';this.src='fixture.png'}
  append(...nodes){this.children.push(...nodes)} replaceChildren(...nodes){this.children=nodes}
  insertBefore(node){this.children.push(node)} setAttribute(){} scrollIntoView(){} click(){}
}
const nodes=new Map(),events={};
const document={createElement:()=>new Element(),getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id)},
  querySelector:id=>document.getElementById(id),querySelectorAll:()=>[],addEventListener:(name,handler)=>events[name]=handler};
const ctx=vm.createContext({document,console,result:null,lastInput:null,busy:()=>{},get:()=>{},URL:{},Blob:class{}});
vm.runInContext(source,ctx);
const grid=x=>Array.from({length:4},()=>Array(4).fill(x));
function fixture({below=false,refused=false,uncertain=false}={}){
  const target={image_id:'target',score:below?.2:.4,accepted:!below,
    contributions:below?{global:.4,local_64:-.1,local_192:-.1}:{global:.5,local_64:-.1,local_192:0}};
  const competitor={image_id:'competitor',score:refused?.3:below?.5:.3,accepted:below&&!refused,
    contributions:below&&!refused?{global:.3,local_64:.15,local_192:.05}:{global:.2,local_64:.1,local_192:0}};
  const proof={input_tensor_sha256:'input',query_embedding_sha256:'embedding',gallery_sha256:'gallery',weights_manifest_sha256:'model',gallery_size:750};
  const search={search_token:'current',provenance:proof,gallery_size:750,threshold:.35,refused,
    ranking:below?[competitor,target]:[target,competitor]};
  const arm=()=>({changed_winner:grid(false),changed_winner_uncertain:grid(false),competitor_id:grid('competitor'),
    target_margin_delta:grid(.01),target_margin_signal_above_noise:grid(true)});
  const data={target_id:'target',provenance:{...proof,search_token:'current',verified_same_search:true},
    baseline:{target_score:target.score,competitor_score:competitor.score,competitor_id:'competitor',margin:target.score-competitor.score,
      threshold:.35,target_accepted:target.accepted,target_acceptance_uncertain:uncertain,overall_refused:refused,overall_refusal_uncertain:uncertain},
    arms:{gray:arm(),gaussian_blur:arm()},agreement:{margin_delta_comparable_count:8,margin_delta_same_sign_count:5,
      margin_delta_same_sign:grid(true),margin_delta_comparable:grid(true)},
    controls:{full_gallery_rescored_for_every_variant:true,effect_signal_boundary:1e-6},seconds:1,status:refused?'refused':'explained'};
  data.arms.gray.changed_winner[0]=[true,true,true,false];data.arms.gray.changed_winner_uncertain[0][2]=true;
  data.arms.gaussian_blur.changed_winner[0][0]=true;data.arms.gaussian_blur.competitor_id[0][0]='new';
  return {search,data};
}
let passed=0;function test(name,fn){fn();passed++;console.log('PASS '+name)}
const build=({search,data},token='current')=>ctx.buildDecisionEvidence(search,data,token);
test('Signed branch opposition and distinct gray/blur changes',()=>{
  const e=build(fixture());assert.equal(e.branch_conflict,true);assert.equal(e.interventions.gray.definite_winner_changes,2);
  assert.equal(e.interventions.gray.uncertain_winner_changes,1);assert.equal(e.interventions.gaussian_blur.definite_winner_changes,1);
  assert.equal(e.interventions.gaussian_blur.competitor_changes,1);assert.equal(e.agreement.disagree,3);assert.equal(e.agreement.not_comparable,8);
  assert(Math.abs(e.threshold_gap-.05)<1e-12);assert.equal(e.branch_contributions[1].margin,-.2);
});
test('Below-threshold candidate does not turn an accepted search into refusal',()=>{
  const e=build(fixture({below:true}));assert.equal(e.accepted,false);assert.equal(e.overall_refused,false);assert.equal(e.rank,2);
  assert(e.paragraphs.some(x=>x.includes('другой результат принят')));assert(!e.paragraphs.some(x=>x.includes('поиск отказался')));
});
test('True overall refusal is distinguished and numerical uncertainty disclosed',()=>{
  const e=build(fixture({below:true,refused:true}));assert(e.paragraphs.some(x=>x.includes('поиск отказался')));
  const u=build(fixture({uncertain:true}));assert(u.paragraphs.some(x=>x.includes('численно неустойчив')));
  assert(u.actions.some(x=>x.includes('численной границы')));
});
test('Non-comparable effects are not agreement or proof of robustness',()=>{
  const f=fixture();f.data.agreement.margin_delta_comparable_count=0;f.data.agreement.margin_delta_same_sign_count=0;
  const e=build(f);assert.equal(e.agreement.agree,0);assert.equal(e.agreement.not_comparable,16);
  assert(e.paragraphs.some(x=>x.includes('не доказательство устойчивости')));
});
test('Token, model, gallery, embedding and input mismatches fail closed',()=>{
  assert.throws(()=>build(fixture(),'new-token'));
  for(const key of ['search_token','input_tensor_sha256','query_embedding_sha256','gallery_sha256','weights_manifest_sha256','gallery_size']){
    const f=fixture();f.data.provenance[key]='changed';assert.throws(()=>build(f));
  }
});
test('Unknown candidate, wrong decision and inconsistent branch arithmetic reject',()=>{
  const f=fixture();f.data.target_id='missing';assert.throws(()=>build(f));
  const g=fixture();g.data.baseline.overall_refused=true;assert.throws(()=>build(g));
  const h=fixture();h.search.ranking[0].contributions.global+=.1;assert.throws(()=>build(h));
});
test('Export binds full current ranking, token, hashes and original explanation',()=>{
  const f=fixture(),e=build(f),out=JSON.parse(JSON.stringify(ctx.evidenceExport(f.search,f.data,e)));
  assert.equal(out.search_token,'current');assert.equal(out.search_snapshot.provenance.gallery_sha256,'gallery');
  assert.equal(out.search_snapshot.ranking.length,2);assert.equal(out.explanation.provenance.search_token,'current');
  assert.equal(out.operator_summary.schema,'falcon.operator-evidence.v1');
  const markdown=ctx.evidenceMarkdown(out);assert(markdown.includes('gallery_sha256: gallery'));
  assert(markdown.includes('оригинальные вход, модель и галерея'));assert(!markdown.includes('<img'));
  assert(markdown.includes('| Локальные 64 |'));
});
test('Rendered summary invalidates at search start, stale export cannot remain',()=>{
  const f=fixture();ctx.result=f.search;ctx.lastInput={search_token:'current'};
  ctx.renderExplanation(f.data,'query.png','target');assert.equal(vm.runInContext('explanationEvidence.search_token',ctx),'current');
  assert(document.getElementById('decision-evidence').children.length>5);
  events['falcon-search-start']();assert.equal(vm.runInContext('explanationEvidence',ctx),null);
  assert.equal(vm.runInContext('explanationResult',ctx),null);assert.equal(vm.runInContext('explanationSection.hidden',ctx),true);
  ctx.lastInput={search_token:'new-token'};assert.throws(()=>ctx.renderExplanation(f.data,'old.png','target'));
});
test('Wrong target response cannot label another image as explained',()=>{
  const f=fixture();ctx.result=f.search;ctx.lastInput={search_token:'current'};
  assert.throws(()=>ctx.renderExplanation(f.data,'query.png','different-target'));
});
test('New search immediately clears old cards, decision and query map',()=>{
  const html=fs.readFileSync(__dirname+'/../index.html','utf8');
  assert(html.includes('Диагностический рейтинг до отбора по порогу'));
  assert(html.includes('не считаются найденными совпадениями'));
  const start=html.slice(html.indexOf('function beginSearch(){'),html.indexOf('\nfunction show('));
  ctx.$=document.getElementById;ctx.searchSerial=0;ctx.CustomEvent=class{constructor(type,options){this.type=type;this.detail=options.detail}};
  document.dispatchEvent=event=>events[event.type]?.(event);
  document.getElementById('cards').append(new Element());
  vm.runInContext(start,ctx);vm.runInContext('beginSearch()',ctx);
  assert.equal(document.getElementById('cards').children.length,0);
  assert.equal(document.getElementById('decision').textContent,'Выполняется новый поиск');
  assert.equal(document.getElementById('query').hidden,true);assert.equal(ctx.result,null);
});
console.log(JSON.stringify({passed,scope:'synthetic JS evidence and DOM fixtures; no real model/browser/quality claim'}));
