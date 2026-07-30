const labels = [
  ["confirmed_overlap", "1 Confirmed overlap"], ["likely_overlap", "2 Likely overlap"],
  ["adjacent_no_overlap", "3 Adjacent, no overlap"], ["same_region_no_overlap", "4 Same region, no overlap"],
  ["not_overlap", "5 Not Overlapped"], ["uncertain", "6 Uncertain"],
];
let items = [], active = 0;
const q = id => document.getElementById(id);
async function json(url, options) { const r = await fetch(url, options); if (!r.ok) throw new Error((await r.json()).detail || r.statusText); return r.json(); }
function protocolQuery() { return `protocol=${encodeURIComponent(q("protocol").value)}`; }
function actionFor(row, label) {
  if (label === "uncertain") return "needs_second_review";
  if (["confirmed_overlap", "likely_overlap"].includes(label) && row.split_pair === "train-test") return "exclude_train_side";
  return "keep_both_record_only";
}
function imageUrl(row, side, kind) { return `/api/image/${row.pair_id}/${side}/${kind}?${protocolQuery()}`; }
function sidePanel(row, side) {
  const split = row[`split_${side}`], id = row[`id_${side}`], event = row[`event_${side}`];
  return `<section class="side side-${split}"><h3>${split.toUpperCase()} · ${event} · <code>${id}</code></h3>
    <figure><img loading="lazy" src="${imageUrl(row, side, "pre")}"><figcaption>pre optical</figcaption></figure>
    <details><summary>SAR / mask / overlay</summary><div class="modalities">
      ${["sar","mask","overlay"].map(kind => `<figure><img loading="lazy" src="${imageUrl(row, side, kind)}"><figcaption>${kind}</figcaption></figure>`).join("")}
    </div></details></section>`;
}
function card(row, index) {
  const selected = row.review_label || "";
  return `<article class="candidate ${index === active ? "active" : ""}" id="pair-${row.pair_id}" data-index="${index}">
    <div class="meta"><span class="badge ${row.risk_level}">${row.risk_level}</span> ${row.split_pair} · score=${row.overlap_score} · patch=${row.patch_overlap_score} · inliers=${row.local_inliers}/${row.local_matches}</div>
    <div class="comparison">${sidePanel(row,"a")}${sidePanel(row,"b")}</div>
    <div class="decision-grid">${labels.map(([value,text]) => `<button class="label ${value === selected ? "selected" : ""}" data-label="${value}">${text}</button>`).join("")}</div>
    <div class="review-row"><input class="comment" value="${(row.review_comment || "").replaceAll('"','&quot;')}" placeholder="复核说明（可选）"><span>${row.review_status}</span></div>
  </article>`;
}
function bindCards() {
  document.querySelectorAll("article.candidate").forEach(article => {
    article.onclick = () => { active = Number(article.dataset.index); markActive(false); };
    article.querySelectorAll("button.label").forEach(button => button.onclick = async event => {
      event.stopPropagation(); active = Number(article.dataset.index);
      await save(items[active], button.dataset.label, article.querySelector(".comment").value);
    });
  });
}
function markActive(scroll = true) {
  document.querySelectorAll("article.candidate").forEach((node,index) => node.classList.toggle("active", index === active));
  if (scroll && items[active]) q(`pair-${items[active].pair_id}`).scrollIntoView({behavior:"smooth",block:"start"});
}
async function save(row, label, comment) {
  const saved = await json(`/api/candidates/${row.pair_id}/decision?${protocolQuery()}`, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({review_label:label, review_action:actionFor(row,label), comment}),
  });
  items[active] = saved;
  q(`pair-${row.pair_id}`).outerHTML = card(saved, active);
  bindCards(); markActive(false); await loadSummary();
}
async function loadSummary() {
  const s = await json(`/api/summary?${protocolQuery()}`);
  q("summary").innerHTML = `<p><b>${s.protocol}</b><br>候选 ${s.total}<br>必审 train-test ${s.mandatory_train_test.reviewed}/${s.mandatory_train_test.total}</p>`;
  q("riskStats").innerHTML = ["high","medium","low"].map(risk => `<button data-risk="${risk}"><b>${risk.toUpperCase()}</b><span>${s.risk[risk] || 0}</span><small>未审 ${s.risk_review_status[risk]?.unreviewed || 0}</small></button>`).join("");
  q("riskStats").querySelectorAll("button").forEach(button => button.onclick = () => { q("risk").value=button.dataset.risk; load(); });
  q("exportManifest").disabled = q("protocol").value !== "strict_v1" || s.mandatory_train_test.unreviewed > 0;
}
async function load() {
  const params = new URLSearchParams({protocol:q("protocol").value, split_pair:q("splitPair").value, risk:q("risk").value, status:q("status").value, limit:"500"});
  const data = await json(`/api/candidates?${params}`); items = data.items; active = 0;
  q("candidateList").innerHTML = items.map(card).join(""); q("empty").hidden = items.length > 0;
  bindCards(); await loadSummary();
  q("notice").textContent = q("protocol").value === "legacy" ? "Legacy 对照：每组上方为原 Train，下方为原 Test。" : "Strict-v1：Not Overlapped 会保留双方；只有显式导出才生成新的 reviewed manifest。";
}
q("reload").onclick = load; q("protocol").onchange = load;
q("exportManifest").onclick = async () => {
  if (!confirm("生成新的 strict-v1 overlap-reviewed manifest？不会覆盖原 strict-v1。")) return;
  try { const result=await json(`/api/apply_decisions?${protocolQuery()}`,{method:"POST"}); alert(`${result.status}: ${result.out_root}`); }
  catch(error) { alert(error.message); }
};
document.addEventListener("keydown", async event => {
  if (["INPUT","SELECT","TEXTAREA"].includes(document.activeElement.tagName) || !items.length) return;
  if (event.key.toLowerCase() === "j") { active=Math.min(items.length-1,active+1); markActive(); }
  else if (event.key.toLowerCase() === "k") { active=Math.max(0,active-1); markActive(); }
  else if (/^[1-6]$/.test(event.key)) { const label=labels[Number(event.key)-1][0]; await save(items[active],label,""); }
});
load().catch(error => { q("notice").textContent=error.message; });
