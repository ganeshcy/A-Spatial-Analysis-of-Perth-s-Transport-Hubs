"""
Build the self-contained HRI explorer page.

Injects output/web/nodes.json into a single HTML document with no external
dependencies: no map library, no tile server, no CDN. The basemap is the stop
pattern itself — 11,773 nodes trace Perth's coastline, river and rail corridors
without needing tiles behind them.

Usage:  python3 build_map.py
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(BASE, "output", "web", "nodes.json")
BASEMAP = os.path.join(BASE, "output", "web", "basemap.json")
DEST = os.path.join(os.path.dirname(BASE), os.pardir, "hri-explorer-site", "index_offline.html")

payload = json.load(open(DATA))
meta = payload["meta"]
basemap = json.load(open(BASEMAP)) if os.path.exists(BASEMAP) else {
    "coast": [], "water": [], "park": [], "edges": []}

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perth Hub Reliance Index</title>
<style>
  :root{ color-scheme: light;
    --ink:#0F1E3D; --muted:#64748B; --line:#E2E8F0; --bg:#FBFCFD;
    --panel:#FFFFFF; --accent:#0E7C86; --warm:#C2410C;
  }
  *{box-sizing:border-box}
  #hri-app{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    color:var(--ink); background:var(--bg); border:1px solid var(--line);
    border-radius:10px; overflow:hidden; display:flex; flex-direction:column; height:760px}
  .hd{padding:12px 16px; border-bottom:1px solid var(--line); background:var(--panel);
    display:flex; align-items:baseline; gap:14px; flex-wrap:wrap}
  .hd h2{margin:0; font-size:15px; font-weight:650; letter-spacing:-.01em}
  .hd .sub{font-size:11.5px; color:var(--muted)}
  .body{flex:1; display:flex; min-height:0}
  .map{flex:1; position:relative; background:#F4F7FA; min-width:0}
  canvas{display:block; width:100%; height:100%; cursor:crosshair}
  .side{width:308px; border-left:1px solid var(--line); background:var(--panel);
    display:flex; flex-direction:column; min-height:0}
  .ctl{padding:11px 13px; border-bottom:1px solid var(--line)}
  .ctl label{display:block; font-size:10.5px; font-weight:600; text-transform:uppercase;
    letter-spacing:.05em; color:var(--muted); margin-bottom:5px}
  select,input[type=text]{width:100%; padding:6px 8px; font-size:12.5px; font-family:inherit;
    border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink)}
  input[type=range]{width:100%; accent-color:var(--accent); margin:2px 0}
  .brow{display:flex; justify-content:space-between; font-size:11px; color:var(--muted)}
  .bval{font-variant-numeric:tabular-nums; color:var(--accent); font-weight:700; font-size:12.5px}
  .chips{display:flex; flex-wrap:wrap; gap:5px; margin-top:2px}
  .chip{font-size:10.5px; padding:3px 8px; border-radius:20px; border:1px solid var(--line);
    cursor:pointer; user-select:none; background:#fff; display:flex; align-items:center; gap:5px}
  .chip.off{opacity:.35}
  .dot{width:8px;height:8px;border-radius:50%}
  .list{flex:1; overflow-y:auto; min-height:0}
  .item{padding:7px 13px; border-bottom:1px solid #F1F5F9; cursor:pointer; font-size:12px;
    display:flex; gap:9px; align-items:baseline}
  .item:hover{background:#F8FAFC}
  .item .rk{font-variant-numeric:tabular-nums; color:var(--muted); font-size:10.5px; width:22px; flex:none}
  .item .nm{flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
  .item .sb{font-size:10px; color:var(--muted)}
  .item .vl{font-variant-numeric:tabular-nums; font-weight:650; color:var(--accent); font-size:11.5px}
  .det{border-top:1px solid var(--line); padding:11px 13px; background:#FCFDFE; max-height:246px; overflow-y:auto}
  .det h3{margin:0 0 1px; font-size:13px; font-weight:650}
  .det .sb2{font-size:11px; color:var(--muted); margin-bottom:8px}
  .kv{display:grid; grid-template-columns:1fr auto; gap:2px 10px; font-size:11.5px}
  .kv .k{color:var(--muted)}
  .kv .v{font-variant-numeric:tabular-nums; font-weight:600}
  .bar{height:3px; background:var(--line); border-radius:2px; overflow:hidden; grid-column:1/-1; margin:1px 0 4px}
  .bar i{display:block; height:100%; background:var(--accent)}
  .lg{position:absolute; left:12px; bottom:12px; background:rgba(255,255,255,.95);
    border:1px solid var(--line); border-radius:7px; padding:8px 10px; font-size:10.5px}
  .lg .ramp{height:8px; width:150px; border-radius:2px; margin:4px 0 3px}
  .lg .ends{display:flex; justify-content:space-between; color:var(--muted); font-variant-numeric:tabular-nums}
  .hint{position:absolute; right:12px; top:12px; background:rgba(255,255,255,.95);
    border:1px solid var(--line); border-radius:7px; padding:6px 9px; font-size:10.5px; color:var(--muted)}
  .tag{display:inline-block; padding:1px 7px; border-radius:20px; font-size:9.5px;
    font-weight:650; text-transform:uppercase; letter-spacing:.04em; color:#fff}
</style>

</head>
<body style="margin:0;padding:14px;background:#fff">
<div id="hri-app">
  <div class="hd">
    <h2>Perth Hub Reliance Index</h2>
    <span class="sub" id="metaline"></span>
  </div>
  <div class="body">
    <div class="map">
      <canvas id="cv"></canvas>
      <div class="lg">
        <div id="lgTitle" style="font-weight:650"></div>
        <div class="ramp" id="lgRamp"></div>
        <div class="ends"><span id="lgLo"></span><span id="lgHi"></span></div>
      </div>
      <div class="hint">drag to pan &middot; scroll to zoom &middot; click a stop</div>
    </div>
    <div class="side">
      <div class="ctl">
        <label>Colour by</label>
        <select id="varSel"></select>
      </div>
      <div class="ctl">
        <label for="beta">&beta; &mdash; reliance amplification</label>
        <div class="brow"><span>0</span><span class="bval" id="bOut">0.50</span><span>1.5</span></div>
        <input type="range" id="beta" min="0" max="1.5" step="0.05" value="0.5">
        <div class="brow"><span id="bNote"></span></div>
      </div>
      <div class="ctl">
        <label>Typology</label>
        <div class="chips" id="chips"></div>
      </div>
      <div class="ctl">
        <label>Layers</label>
        <div class="chips" id="layers"></div>
      </div>
      <div class="ctl"><input type="text" id="q" placeholder="Search stop or suburb..."></div>
      <div class="list" id="list"></div>
      <div class="det" id="det"></div>
    </div>
  </div>
</div>

<script>
(function(){
const D = __DATA__;
const BM = __BASEMAP__;
const N = D.meta.n;

// HRI is recomputed here rather than shipped, because beta is undecided.
let beta = D.meta.beta_default;
const hri = new Float32Array(N);
const rank = new Int32Array(N);
let order = null;

const TYPES = {
  critical:  {c:'#0F1E3D', label:'critical'},
  captive:   {c:'#C2410C', label:'captive'},
  integrated:{c:'#0E7C86', label:'integrated'},
  redundant: {c:'#94A3B8', label:'redundant'}
};
// typology is derived here too, so it tracks beta-independent S_plus and L
const medS = median(D.s), medL = median(D.l);
const typ = new Array(N);
for(let i=0;i<N;i++){
  const hS = D.s[i] > medS, hL = D.l[i] > medL;
  typ[i] = hS ? (hL?'critical':'integrated') : (hL?'captive':'redundant');
}
function median(a){const b=Float64Array.from(a);b.sort();return b[b.length>>1];}

const VARS = [
  {k:'hri', n:'Hub Reliance Index (HRI)', get:i=>hri[i], d:3},
  {k:'typ', n:'Typology (categorical)', get:i=>0, d:0, cat:true},
  {k:'s', n:'S₊ — supply & position', get:i=>D.s[i], d:3},
  {k:'l', n:'L — reliance', get:i=>D.l[i], d:3},
  {k:'v', n:'V — zero-car share', get:i=>D.v[i], d:3},
  {k:'a', n:'D — catchment demand', get:i=>D.a[i], d:0},
  {k:'g', n:'G — peak delay (seconds)', get:i=>D.g[i], d:1},
  {k:'c', n:'C — centrality', get:i=>D.c[i], d:4},
  {k:'t', n:'T — transfer potential', get:i=>D.t[i], d:0},
  {k:'r', n:'R — routes', get:i=>D.r[i], d:0},
  {k:'f', n:'F — frequency (collated)', get:i=>D.f[i], d:2}
];
let cur = VARS[0];
const active = new Set(Object.keys(TYPES));

// ---------------------------------------------------------------- colour
const RAMP = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function ramp(t){
  t = Math.max(0,Math.min(1,t));
  const x = t*(RAMP.length-1), i = Math.floor(x), f = x-i;
  const a = RAMP[i], b = RAMP[Math.min(i+1,RAMP.length-1)];
  return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+Math.round(a[1]+(b[1]-a[1])*f)
        +','+Math.round(a[2]+(b[2]-a[2])*f)+')';
}
let breaks = [];
function computeBreaks(){
  if(cur.cat){breaks=[];return;}
  const v=[]; for(let i=0;i<N;i++) v.push(cur.get(i));
  v.sort((a,b)=>a-b);
  breaks=[]; for(let q=0;q<=10;q++) breaks.push(v[Math.min(v.length-1,Math.floor(q/10*(v.length-1)))]);
}
function colOf(i){
  if(cur.cat) return TYPES[typ[i]].c;
  const v=cur.get(i); let lo=breaks[0],hi=breaks[breaks.length-1];
  // quantile position, so skewed components still show structure
  let p=0; for(let q=1;q<breaks.length;q++){ if(v<=breaks[q]){p=(q-1+ (breaks[q]>breaks[q-1]?(v-breaks[q-1])/(breaks[q]-breaks[q-1]):0))/10; break;} p=1; }
  return ramp(p);
}

// ------------------------------------------------------------ projection
const R2D = Math.PI/180;
const mx = new Float64Array(N), my = new Float64Array(N);
for(let i=0;i<N;i++){
  mx[i] = D.lon[i]*R2D;
  my[i] = Math.log(Math.tan(Math.PI/4 + D.lat[i]*R2D/2));
}
let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
for(let i=0;i<N;i++){ if(mx[i]<x0)x0=mx[i]; if(mx[i]>x1)x1=mx[i];
                      if(my[i]<y0)y0=my[i]; if(my[i]>y1)y1=my[i]; }

// Basemap rings share the stop projection exactly, so nothing can drift.
function projRings(rs){
  return (rs||[]).map(r=>{
    const out=new Float64Array(r.length*2);
    for(let k=0;k<r.length;k++){
      out[k*2]   = r[k][0]*R2D;
      out[k*2+1] = Math.log(Math.tan(Math.PI/4 + r[k][1]*R2D/2));
    }
    return out;
  });
}
const BMP = {coast:projRings(BM.coast), water:projRings(BM.water), park:projRings(BM.park)};
const EDG = BM.edges||[];
const layers = {basemap:true, network:false};

const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let W=0,H=0,dpr=1, scale=1, panX=0, panY=0, base=1, bx=0, by=0;
function resize(){
  const r=cv.getBoundingClientRect(); dpr=window.devicePixelRatio||1;
  W=r.width; H=r.height; cv.width=W*dpr; cv.height=H*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const pad=26;
  base=Math.min((W-2*pad)/(x1-x0),(H-2*pad)/(y1-y0));
  bx=(W-(x1-x0)*base)/2; by=(H-(y1-y0)*base)/2;
  draw();
}
function sx(i){ return (bx+(mx[i]-x0)*base)*scale+panX; }
function sy(i){ return (by+(y1-my[i])*base)*scale+panY; }

let sel=-1;
function ringPath(fr){
  ctx.beginPath();
  for(let k=0;k<fr.length;k+=2){
    const X=(bx+(fr[k]-x0)*base)*scale+panX, Y=(by+(y1-fr[k+1])*base)*scale+panY;
    if(k===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y);
  }
  ctx.closePath();
}
function drawBasemap(){
  // ocean first, then land, then parks and water on top of it
  ctx.fillStyle='#E8EFF5'; ctx.fillRect(0,0,W,H);
  ctx.fillStyle='#FAFBFC'; ctx.strokeStyle='#DCE3EA'; ctx.lineWidth=1;
  BMP.coast.forEach(fr=>{ ringPath(fr); ctx.fill(); ctx.stroke(); });
  ctx.fillStyle='#EDF3EC';
  BMP.park.forEach(fr=>{ ringPath(fr); ctx.fill(); });
  ctx.fillStyle='#D6E6F2';
  BMP.water.forEach(fr=>{ ringPath(fr); ctx.fill(); });
}
function drawEdges(){
  ctx.strokeStyle='#C3CEDA'; ctx.lineWidth=Math.min(1.1,0.35*Math.pow(scale,0.4));
  ctx.globalAlpha=0.55; ctx.beginPath();
  for(let e=0;e<EDG.length;e+=2){
    const a=EDG[e], b=EDG[e+1];
    const ax=sx(a), ay=sy(a); if(ax<-60||ax>W+60||ay<-60||ay>H+60) continue;
    ctx.moveTo(ax,ay); ctx.lineTo(sx(b),sy(b));
  }
  ctx.stroke(); ctx.globalAlpha=1;
}
function draw(){
  ctx.clearRect(0,0,W,H);
  if(layers.basemap) drawBasemap(); else { ctx.fillStyle='#F4F7FA'; ctx.fillRect(0,0,W,H); }
  if(layers.network) drawEdges();
  const r=Math.max(1.1, 1.5*Math.pow(scale,0.45));
  // draw inactive first so active types sit on top
  for(let pass=0;pass<2;pass++){
    for(let i=0;i<N;i++){
      const on=active.has(typ[i]); if(pass===0? on : !on) continue;
      const X=sx(i), Y=sy(i); if(X<-5||X>W+5||Y<-5||Y>H+5) continue;
      ctx.globalAlpha = on?0.86:0.07;
      ctx.fillStyle = on?colOf(i):'#94A3B8';
      ctx.beginPath(); ctx.arc(X,Y,r,0,6.283); ctx.fill();
    }
  }
  ctx.globalAlpha=1;
  if(sel>=0){
    const X=sx(sel),Y=sy(sel);
    ctx.strokeStyle='#0F1E3D'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(X,Y,Math.max(7,r+5),0,6.283); ctx.stroke();
    ctx.strokeStyle='#fff'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.arc(X,Y,Math.max(7,r+5)+1.5,0,6.283); ctx.stroke();
  }
}

// ------------------------------------------------------------ interaction
let drag=false,lx=0,ly=0;
cv.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;});
window.addEventListener('mouseup',()=>{drag=false;});
window.addEventListener('mousemove',e=>{
  if(!drag)return; panX+=e.clientX-lx; panY+=e.clientY-ly; lx=e.clientX; ly=e.clientY; draw();
});
cv.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=cv.getBoundingClientRect(), px=e.clientX-r.left, py=e.clientY-r.top;
  const f=e.deltaY<0?1.18:1/1.18, ns=Math.max(1,Math.min(90,scale*f));
  panX=px-(px-panX)*(ns/scale); panY=py-(py-panY)*(ns/scale); scale=ns; draw();
},{passive:false});
cv.addEventListener('click',e=>{
  const r=cv.getBoundingClientRect(), px=e.clientX-r.left, py=e.clientY-r.top;
  let best=-1,bd=225;
  for(let i=0;i<N;i++){
    if(!active.has(typ[i]))continue;
    const dx=sx(i)-px, dy=sy(i)-py, d=dx*dx+dy*dy;
    if(d<bd){bd=d;best=i;}
  }
  if(best>=0){ sel=best; showDetail(best); draw(); }
});

// ------------------------------------------------------------------ recompute
function recompute(){
  for(let i=0;i<N;i++) hri[i]=D.s[i]*(1+beta*D.l[i]);
  order=Array.from({length:N},(_,i)=>i).sort((a,b)=>hri[b]-hri[a]);
  for(let k=0;k<N;k++) rank[order[k]]=k+1;
  computeBreaks(); renderList(); updateLegend();
  if(sel>=0) showDetail(sel);
  draw();
}

// ------------------------------------------------------------------- UI
const listEl=document.getElementById('list'), detEl=document.getElementById('det');
const qEl=document.getElementById('q');
function renderList(){
  const q=qEl.value.trim().toLowerCase();
  const rows=[];
  for(let k=0;k<N && rows.length<150;k++){
    const i=order[k];
    if(!active.has(typ[i]))continue;
    if(q && !(D.name[i].toLowerCase().includes(q)||D.sub[i].toLowerCase().includes(q)))continue;
    rows.push('<div class="item" data-i="'+i+'"><span class="rk">'+(k+1)+'</span>'
      +'<span class="nm">'+esc(D.name[i])+'<br><span class="sb">'+esc(D.sub[i])+'</span></span>'
      +'<span class="vl">'+hri[i].toFixed(3)+'</span></div>');
  }
  listEl.innerHTML=rows.join('')||'<div style="padding:16px;font-size:12px;color:#64748B">No stops match.</div>';
  listEl.querySelectorAll('.item').forEach(el=>el.onclick=()=>{
    const i=+el.dataset.i; sel=i; centreOn(i); showDetail(i); draw();
  });
}
function centreOn(i){
  if(scale<7) scale=7;
  panX = W/2 - (bx+(mx[i]-x0)*base)*scale;
  panY = H/2 - (by+(y1-my[i])*base)*scale;
}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function showDetail(i){
  const t=TYPES[typ[i]];
  const pct=v=>Math.max(0,Math.min(1,v))*100;
  detEl.innerHTML =
    '<h3>'+esc(D.name[i])+'</h3>'
   +'<div class="sb2">'+esc(D.sub[i])+' &middot; id '+D.id[i]
   +' &middot; <span class="tag" style="background:'+t.c+'">'+t.label+'</span></div>'
   +'<div class="kv">'
   +'<span class="k">HRI</span><span class="v">'+hri[i].toFixed(3)+'  (rank '+rank[i]+')</span>'
   +'<div class="bar"><i style="width:'+pct(hri[i]/2.5)+'%"></i></div>'
   +'<span class="k">S₊ supply &amp; position</span><span class="v">'+D.s[i].toFixed(3)+'</span>'
   +'<div class="bar"><i style="width:'+pct(D.s[i])+'%"></i></div>'
   +'<span class="k">L reliance</span><span class="v">'+D.l[i].toFixed(3)+'</span>'
   +'<div class="bar"><i style="width:'+pct(D.l[i])+'%;background:#C2410C"></i></div>'
   +'<span class="k">V zero-car dwellings</span><span class="v">'+(D.v[i]*100).toFixed(1)+'%</span>'
   +'<span class="k">D catchment demand</span><span class="v">'+D.a[i].toLocaleString()+'</span>'
   +'<span class="k">R routes</span><span class="v">'+D.r[i]+'</span>'
   +'<span class="k">T transfer potential</span><span class="v">'+D.t[i]+'</span>'
   +'<span class="k">C centrality</span><span class="v">'+D.c[i].toFixed(4)+'</span>'
   +'<span class="k">G peak delay</span><span class="v">'+(D.g[i]?('+'+D.g[i].toFixed(0)+' s'):'\u2014')+'</span>'
   +'<span class="k">alternatives nearby (A)</span><span class="v">'+D.alt[i]+'</span>'
   +'<span class="k">departures AM / off / PM / rem</span><span class="v">'
     +D.am[i]+' / '+D.off[i]+' / '+D.pm[i]+' / '+D.rem[i]+'</span>'
   +'</div>';
}
function updateLegend(){
  document.getElementById('lgTitle').textContent=cur.n;
  const rp=document.getElementById('lgRamp');
  if(cur.cat){
    rp.style.background='linear-gradient(90deg,'+Object.values(TYPES).map(t=>t.c).join(',')+')';
    document.getElementById('lgLo').textContent='critical / captive';
    document.getElementById('lgHi').textContent='integrated / redundant';
  }else{
    rp.style.background='linear-gradient(90deg,'+[0,.25,.5,.75,1].map(t=>ramp(t)).join(',')+')';
    document.getElementById('lgLo').textContent=fmt(breaks[0],cur.d);
    document.getElementById('lgHi').textContent=fmt(breaks[breaks.length-1],cur.d);
  }
}
function fmt(v,d){return d===0?Math.round(v).toLocaleString():v.toFixed(d);}

const sel1=document.getElementById('varSel');
VARS.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v.n;sel1.appendChild(o);});
sel1.onchange=()=>{cur=VARS[+sel1.value];computeBreaks();updateLegend();draw();};

const chips=document.getElementById('chips');
Object.entries(TYPES).forEach(([k,t])=>{
  const c=document.createElement('div'); c.className='chip';
  c.innerHTML='<span class="dot" style="background:'+t.c+'"></span>'+t.label;
  c.onclick=()=>{ if(active.has(k))active.delete(k); else active.add(k);
    c.classList.toggle('off',!active.has(k)); renderList(); draw(); };
  chips.appendChild(c);
});

const LAYERS=[['basemap','Coast, river, parks'],['network','Network edges']];
const layEl=document.getElementById('layers');
LAYERS.forEach(([k,label])=>{
  const c=document.createElement('div');
  c.className='chip'+(layers[k]?'':' off'); c.textContent=label;
  c.onclick=()=>{ layers[k]=!layers[k]; c.classList.toggle('off',!layers[k]); draw(); };
  layEl.appendChild(c);
});

const bEl=document.getElementById('beta'), bOut=document.getElementById('bOut'),
      bNote=document.getElementById('bNote');
bEl.oninput=()=>{
  beta=+bEl.value; bOut.textContent=beta.toFixed(2);
  bNote.textContent = beta===0 ? 'reliance off — supply-side ranking only'
    : (beta<0.35?'below the stability band':(beta<=1.0?'inside the stability band [0.35, 1.00]':'above the stability band'));
  recompute();
};
qEl.oninput=renderList;

document.getElementById('metaline').textContent =
  N.toLocaleString()+' merged nodes · CRITIC weights ('+D.meta.weight_frame
  +' frame) · T at '+D.meta.t_threshold+' · HRI = S₊ × (1 + β · L)';

new ResizeObserver(resize).observe(cv.parentElement);
bEl.value=beta; bOut.textContent=beta.toFixed(2); bEl.oninput();
resize();
})();
</script>
</body></html>
"""

out = (HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
           .replace("__BASEMAP__", json.dumps(basemap, separators=(",", ":"))))
os.makedirs(os.path.dirname(DEST), exist_ok=True)
open(DEST, "w").write(out)
print(f"wrote {DEST}  {os.path.getsize(DEST)/1e6:.2f} MB")
print(f"  {meta['n']:,} nodes, beta default {meta['beta_default']}, "
      f"frame {meta['weight_frame']}, T {meta['t_threshold']}")
