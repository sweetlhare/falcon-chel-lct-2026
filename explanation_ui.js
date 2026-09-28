// Deterministic wording from measured results; no identity or semantic inference.
function buildDecisionEvidence(search,data,currentToken){
  const fail=()=>{throw new Error('Проверка не соответствует текущему поиску. Повторите поиск и объяснение.')};
  const proof=data.provenance;
  if(!currentToken||search.search_token!==currentToken||!proof?.verified_same_search||proof.search_token!==currentToken)fail();
  for(const key of ['input_tensor_sha256','query_embedding_sha256','gallery_sha256','weights_manifest_sha256','gallery_size']){
    if(search.provenance?.[key]===undefined||search.provenance[key]!==proof[key])fail();
  }
  const b=data.baseline,target=search.ranking.find(x=>x.image_id===data.target_id);
  const competitor=search.ranking.find(x=>x.image_id===b.competitor_id);
  if(!target||!competitor||Math.abs(target.score-b.target_score)>1e-4||
     Math.abs(competitor.score-b.competitor_score)>1e-4||search.threshold!==b.threshold||
     search.refused!==b.overall_refused||data.controls.full_gallery_rescored_for_every_variant!==true)fail();
  const blocks=[['global','Общий вид'],['local_64','Локальные 64'],['local_192','Локальные 192']].map(([key,label])=>({
    key,label,target:target.contributions[key],competitor:competitor.contributions[key],
    margin:target.contributions[key]-competitor.contributions[key]}));
  const noise=data.controls.effect_signal_boundary;
  if(!Number.isFinite(noise)||blocks.some(x=>![x.target,x.competitor,x.margin].every(Number.isFinite)))fail();
  if(Math.abs(blocks.reduce((s,x)=>s+x.target,0)-target.score)>1e-6||
     Math.abs(blocks.reduce((s,x)=>s+x.competitor,0)-competitor.score)>1e-6)fail();
  const interventions={};
  for(const key of ['gray','gaussian_blur']){
    const arm=data.arms[key],changed=arm.changed_winner.flat(),uncertain=arm.changed_winner_uncertain.flat();
    interventions[key]={total:changed.length,winner_changes:changed.filter(Boolean).length,
      definite_winner_changes:changed.filter((x,i)=>x&&!uncertain[i]).length,
      uncertain_winner_changes:changed.filter((x,i)=>x&&uncertain[i]).length,
      competitor_changes:arm.competitor_id.flat().filter(id=>id!==b.competitor_id).length};
  }
  const comparable=data.agreement.margin_delta_comparable_count,agree=data.agreement.margin_delta_same_sign_count;
  const agreement={comparable,agree,disagree:comparable-agree,not_comparable:interventions.gray.total-comparable};
  const rank=search.ranking.findIndex(x=>x.image_id===data.target_id)+1;
  const fmt=x=>Number(x).toFixed(5),signed=x=>(x>=0?'+':'')+fmt(x);
  const paragraphs=[`Кандидат №${rank}: оценка ${fmt(target.score)}, порог ${fmt(search.threshold)}, разность ${signed(target.score-search.threshold)}. `+
    (b.target_acceptance_uncertain?'Решение о принятии этого кандидата находится в пределах численного порога.':
      target.accepted?'Этот кандидат принят по порогу.':'Этот кандидат не принят по порогу.')];
  paragraphs.push(b.overall_refusal_uncertain?'Пороговый итог всего поиска численно неустойчив.':
    search.refused?'Во всей галерее нет кандидатов выше порога: поиск отказался от совпадения.':
      !target.accepted?'Этот кандидат ниже порога, но другой результат принят: это не общий отказ поиска.':
      'В галерее есть результаты выше порога; это не подтверждение истинного совпадения.');
  paragraphs.push(`Сильнейший конкурент во всей галерее: ${b.competitor_id}. Разница оценок выбранного кандидата и конкурента ${signed(target.score-competitor.score)}. На каждом вмешательстве конкурент выбирается заново.`);
  const branchConflict=blocks.some(x=>x.margin>noise)&&blocks.some(x=>x.margin < -noise);
  if(branchConflict)paragraphs.push('Ветви расходятся: одни поддерживают выбранного кандидата относительно конкурента, другие поддерживают конкурента. Знак вклада виден в таблице.');
  paragraphs.push(comparable?`Из ${comparable} областей, где оба изменения измеримы, направления совпали в ${agree}, разошлись в ${agreement.disagree}. В остальных ${agreement.not_comparable} областях хотя бы один эффект не превышает численный порог.`:
    'Нет областей, где оба способа изменения дают измеримый эффект. Это отсутствие измеримого сигнала в этой проверке, а не доказательство устойчивости личности.');
  const actions=[];
  if(search.refused||!target.accepted)actions.push('Не подтверждайте совпадение по одному этому кандидату: сравните другой кадр автомобиля или оставьте результат неопределённым.');
  if(rank>1)actions.push('Сопоставьте выбранный снимок с первым результатом поиска: он имеет более высокую оценку.');
  if(branchConflict||agreement.disagree)actions.push('Сравните запрос и конкурента рядом; проверьте области с расходящимися направлениями. Их нельзя считать однозначным основанием выбора.');
  if(Object.values(interventions).some(x=>x.definite_winner_changes))actions.push('Лидер меняется при части вмешательств. Проверьте, какие области это вызывают, прежде чем опираться на первый результат.');
  if(b.target_acceptance_uncertain||b.overall_refusal_uncertain||Object.values(interventions).some(x=>x.uncertain_winner_changes))actions.push('Часть решений находится у численной границы. Не трактуйте их как уверенную смену ответа.');
  if(!actions.length)actions.push('Сверьте изображения вручную. Отсутствие выявленных противоречий в этих вмешательствах не подтверждает идентичность автомобиля.');
  return {schema:'falcon.operator-evidence.v1',target_id:data.target_id,rank,score:target.score,
    threshold:search.threshold,threshold_gap:target.score-search.threshold,accepted:target.accepted,
    overall_refused:search.refused,competitor_id:b.competitor_id,margin:target.score-competitor.score,
    branch_contributions:blocks,branch_conflict:branchConflict,interventions,agreement,paragraphs,actions,
    numeric_effect_boundary:noise,provenance:{...proof},
    limitations:['Это чувствительность модели, не доказательство идентичности автомобиля.',
      'Скрытие областей искусственно; эффекты не складываются. Разложение по ветвям относится к исходному поиску.',
      'Автоматическое маскирование номеров неполно. Эта проверка не доказывает отсутствие номерных признаков.']};
}
function evidenceExport(search,data,summary){
  return {schema:'falcon.explanation-evidence.v1',search_token:search.search_token,
    search_snapshot:{provenance:{...search.provenance},threshold:search.threshold,refused:search.refused,
      gallery_size:search.gallery_size,ranking:search.ranking},operator_summary:summary,explanation:data};
}

function evidenceMarkdown(evidence){
  const e=evidence.operator_summary;
  const lines=['# Разбор результата поиска автомобиля','',...e.paragraphs.flatMap(x=>[x,'']),
    '| Ветвь | Кандидат | Конкурент | Разность |','|---|---:|---:|---:|',
    ...e.branch_contributions.map(x=>`| ${x.label} | ${x.target} | ${x.competitor} | ${x.margin} |`),''];
  for(const [key,label] of [['gray','Заливка'],['gaussian_blur','Размытие']]){
    const x=e.interventions[key];lines.push(`${label}: ${x.definite_winner_changes} смен лидера вне численной границы, ${x.uncertain_winner_changes} численно неопределённых; ${x.competitor_changes} смен ID конкурента из ${x.total} проверок.`,'');
  }
  lines.push('## Что проверить','',...e.actions.map(x=>'- '+x),'','## Ограничения','',...e.limitations.map(x=>'- '+x),'',
    '## Привязка к поиску','',`Токен поиска: ${evidence.search_token}`,'',
    ...Object.entries(e.provenance).map(([k,v])=>`${k}: ${v}`),'',
    'Хэши идентифицируют использованные данные. Для повторного расчёта нужны оригинальные вход, модель и галерея; эта карточка их не содержит. Полные численные ответы сохранены в отдельной выгрузке JSON.','');
  return lines.join('\n');
}

// Explain the actual image intervention; never infer a vehicle identity from it.
const explanationSection=document.createElement('section');
explanationSection.className='panel';explanationSection.hidden=true;
explanationSection.innerHTML=`<h2>Что повлияло на выбор</h2>
<p>По очереди скрываем 16 участков кадра и заново сравниваем со всей галереей.
Проверяем два изменения: серую заливку и размытие. Если выводы расходятся, отмечаем область как неустойчивую.</p>
<p id="explanation-status" role="status" aria-live="polite"></p>
<div id="explanation-content" hidden>
<div id="decision-evidence" class="notice" aria-live="polite"></div>
<div class="row"><div><div class="query"><img id="explanation-query" alt="Области проверяемого изображения">
<div id="explanation-map" class="heat" style="grid-template-columns:repeat(4,1fr);pointer-events:auto"></div></div>
<p class="mini">Зелёный: скрытие области уменьшило преимущество кандидата.<br>Оранжевый: скрытие увеличило преимущество.<br>Штриховка: два способа дают разные измеримые направления.</p></div>
<div><img id="explanation-target" width="256" height="256" alt="Анализируемый кандидат"><p id="explanation-target-label" class="mini">Анализируемый кандидат</p></div>
<div><img id="explanation-competitor" width="256" height="256" alt="Ближайший конкурент"><p class="mini">Ближайший конкурент до изменения</p></div></div>
<p id="explanation-summary"></p><div id="explanation-cell" class="notice">Наведите на область или выберите её, чтобы увидеть измеренные изменения.</div>
<details><summary>Числа и границы объяснения</summary><p id="explanation-noise" class="mini"></p><p class="mini">Показываем изменение разницы score с сильнейшим конкурентом, который может меняться после скрытия области. Это проверка поведения модели, а не доказательство, что на снимках один автомобиль. Эффекты областей не складываются; искусственное скрытие может создавать неестественный вход.</p>
<button id="explanation-export">Скачать проверку JSON</button> <button id="explanation-markdown">Скачать разбор .md</button></details></div>`;
document.querySelector('main').insertBefore(explanationSection,document.querySelector('footer'));
let explanationResult=null,explanationEvidence=null,explanationBusy=false,searchVersion=0;
function precise(value){const a=Math.abs(value);if(a===0)return '0';if(a<1e-4)return value.toExponential(2);return value.toFixed(a<.01?6:4)}
function signed(value){return (value>=0?'+':'')+precise(value)}
function explainCell(index){
  const data=explanationResult,r=Math.floor(index/4),c=index%4;
  const a=data.arms.gray,b=data.arms.gaussian_blur;
  const label=`Область ${r+1}:${c+1}. `;
  const describe=(arm,name)=>{
    const value=arm.target_margin_delta[r][c],signal=arm.target_margin_signal_above_noise[r][c];
    const winner=arm.changed_winner[r][c]?(arm.changed_winner_uncertain[r][c]?'возможна смена первого результата в пределах численного порога':'первый результат изменился'):'первый результат сохранился';
    return `${name} ${signed(value)}${signal?'':' (ниже численного порога)'}; ${winner}`;
  };
  document.getElementById('explanation-cell').textContent=label+
    `Изменение после скрытия минус исходное: ${describe(a,'заливка')}. ${describe(b,'Размытие')}.`;
}
function renderExplanation(data,querySource,targetId){
  if(data.target_id!==targetId)throw new Error('Ответ относится к другому кандидату. Повторите проверку.');
  const summary=buildDecisionEvidence(result,data,lastInput?.search_token);
  explanationResult=data;explanationEvidence=evidenceExport(result,data,summary);
  result.explanation=data;result.operator_evidence=summary;
  const evidence=document.getElementById('decision-evidence');evidence.replaceChildren();
  const heading=document.createElement('h3');heading.textContent='Разбор текущего результата';evidence.append(heading);
  for(const text of summary.paragraphs){const p=document.createElement('p');p.textContent=text;evidence.append(p)}
  const table=document.createElement('table');table.style.width='100%';table.style.textAlign='left';
  const caption=document.createElement('caption');caption.textContent='Вклады исходного поиска: положительная разность поддерживает выбранного кандидата';table.append(caption);
  const header=document.createElement('tr');
  for(const label of ['Ветвь','Кандидат','Конкурент','Разность']){const th=document.createElement('th');th.textContent=label;header.append(th)}table.append(header);
  for(const block of [...summary.branch_contributions,{label:'Сумма',target:summary.score,competitor:summary.score-summary.margin,margin:summary.margin}]){
    const row=document.createElement('tr');for(const value of [block.label,precise(block.target),precise(block.competitor),signed(block.margin)]){
      const cell=document.createElement('td');cell.textContent=value;row.append(cell)}table.append(row);
  }evidence.append(table);
  for(const [key,label] of [['gray','Заливка'],['gaussian_blur','Размытие']]){
    const stats=summary.interventions[key],p=document.createElement('p');
    p.textContent=`${label}: первый результат изменился вне численной границы в ${stats.definite_winner_changes} из ${stats.total} проверок; ещё ${stats.uncertain_winner_changes} смен численно неопределённы. Сильнейший конкурент выбранного кандидата сменился в ${stats.competitor_changes} проверках (счётчик по ID, без вывода об устойчивости).`;evidence.append(p);
  }
  const actionHeading=document.createElement('h4');actionHeading.textContent='Что проверить дальше';evidence.append(actionHeading);
  const list=document.createElement('ul');for(const text of summary.actions){const item=document.createElement('li');item.textContent=text;list.append(item)}evidence.append(list);
  const caveat=document.createElement('p');caveat.className='mini';caveat.textContent=summary.limitations.join(' ');evidence.append(caveat);
  document.getElementById('explanation-content').hidden=false;
  document.getElementById('explanation-query').src=querySource;
  document.getElementById('explanation-target').src='/preview/'+encodeURIComponent(targetId);
  document.getElementById('explanation-competitor').src='/preview/'+encodeURIComponent(data.baseline.competitor_id);
  const accepted=data.baseline.target_accepted,acceptanceUncertain=data.baseline.target_acceptance_uncertain;
  document.getElementById('explanation-target-label').textContent=accepted===null?'Анализируемый кандидат':
    acceptanceUncertain?'Анализируемый кандидат · пороговое решение численно неустойчиво':
    accepted?'Анализируемый кандидат · выше порога':'Анализируемый кандидат · ниже порога';
  const map=document.getElementById('explanation-map');map.replaceChildren();
  const first=data.arms.gray.target_margin_delta.flat(),second=data.arms.gaussian_blur.target_margin_delta.flat();
  const agree=data.agreement.margin_delta_same_sign.flat();
  const comparableCells=data.agreement.margin_delta_comparable.flat();
  const noise=data.controls.effect_signal_boundary??Math.max(1e-6,4*(data.controls.baseline_repeat_score_max_abs_error??0));
  const strengths=first.map((x,i)=>(-x-second[i])/2);const max=Math.max(.001,...strengths.map(Math.abs));
  for(let i=0;i<16;i++){
    const cell=document.createElement('button');cell.type='button';cell.style.padding='0';cell.style.borderRadius='0';
    const effect=strengths[i],a=.15+.55*Math.min(1,Math.abs(effect)/max);
    cell.style.background=!comparableCells[i]?'#8797a120':!agree[i]?
      'repeating-linear-gradient(45deg,#92a0b580 0 4px,#11182060 4px 8px)':
      effect>0?`rgba(166,239,120,${a})`:`rgba(255,145,92,${a})`;
    cell.style.border='1px solid #ffffff60';cell.setAttribute('aria-label',`Область ${Math.floor(i/4)+1}:${i%4+1}`);
    cell.onmouseenter=()=>explainCell(i);cell.onfocus=()=>explainCell(i);cell.onclick=()=>explainCell(i);map.append(cell);
  }
  document.getElementById('explanation-summary').textContent='Выберите область, чтобы сопоставить оба измеренных вмешательства. Подробный разбор относится только к текущему поиску.';
  document.getElementById('explanation-noise').textContent=
    `Численный порог эффекта: ±${precise(noise)}. Меньшие raw-изменения сохранены в JSON, но не окрашиваются как сигнал.`;
}
async function explainCandidate(targetId){
  if(explanationBusy||!lastInput||!result)return;
  const input=lastInput;if(!input.search_token){document.getElementById('explanation-status').textContent='Результат поиска не содержит проверочного токена. Повторите поиск.';return}
  explanationBusy=true;busy(true);const version=searchVersion;
  explanationResult=null;explanationEvidence=null;delete result.explanation;delete result.operator_evidence;
  const source=document.getElementById('query').src;
  explanationSection.hidden=false;document.getElementById('explanation-content').hidden=true;
  const state=document.getElementById('explanation-status');state.textContent='Проверяем изменения изображения. Это займёт несколько секунд…';
  explanationSection.scrollIntoView({behavior:'smooth',block:'start'});
  try{
    const data=input.kind==='example'?
      await get('/explain-example/'+encodeURIComponent(input.id)+'?target_id='+encodeURIComponent(targetId)+'&search_token='+encodeURIComponent(input.search_token)):
      await get('/explain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...input.payload,target_id:targetId,search_token:input.search_token})});
    if(version!==searchVersion)return;
    if(!['explained','refused'].includes(data.status))throw new Error('Этот кандидат отсутствует в текущей галерее. Повторите поиск.');
    renderExplanation(data,source,targetId);state.textContent=data.baseline.overall_refused?
      `Проверка завершена за ${data.seconds.toFixed(1)} с. Исходный поиск отказался от совпадения выше порога; карта показывает чувствительность score.`:
      `Проверка завершена за ${data.seconds.toFixed(1)} с. Каждый раз пересчитана вся галерея.`;
  }catch(error){if(version===searchVersion)state.textContent=error.message;}finally{explanationBusy=false;if(version===searchVersion)busy(false);}
}
document.addEventListener('falcon-search-start',()=>{
  searchVersion++;explanationResult=null;explanationEvidence=null;explanationSection.hidden=true;
  document.getElementById('explanation-content').hidden=true;
});
document.addEventListener('falcon-search',event=>{
  searchVersion++;explanationResult=null;explanationEvidence=null;explanationSection.hidden=true;
  [...document.querySelectorAll('#cards .text')].forEach((node,i)=>{
    const button=document.createElement('button');button.type='button';button.textContent='Почему этот кандидат?';button.style.marginTop='12px';button.style.fontSize='13px';
    button.onclick=()=>explainCandidate(event.detail.ranking[i].image_id);node.append(button);
  });
});
document.getElementById('explanation-export').onclick=()=>{
  if(!explanationEvidence||explanationEvidence.search_token!==lastInput?.search_token)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(explanationEvidence,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='falcon-explanation.json';a.click();URL.revokeObjectURL(url);
};

document.getElementById('explanation-markdown').onclick=()=>{
  if(!explanationEvidence||explanationEvidence.search_token!==lastInput?.search_token)return;
  const url=URL.createObjectURL(new Blob([evidenceMarkdown(explanationEvidence)],{type:'text/markdown;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='falcon-explanation.md';a.click();URL.revokeObjectURL(url);
};
