"""
Build the Leaflet edition of the HRI explorer, for public hosting.

Differs from build_map.py in one respect: the basemap is OpenStreetMap tiles
served through Leaflet rather than embedded vector geometry. That is the better
choice for a public static host, where network access is available and a reader
benefits from street-level context. It is NOT usable in the sandboxed artifact
view, which blocks all requests except three named CDNs — build_map.py remains
the edition for that setting.

The stop layer is a custom canvas overlay rather than Leaflet markers. Eleven
thousand CircleMarkers would make panning unusable; one canvas redrawn from
cached projected coordinates stays smooth, and it keeps the beta slider
responsive because recolouring 11,773 points is a single pass.

Usage:  python3 build_map_leaflet.py
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(BASE, "output", "web", "nodes.json")
BASEMAP = os.path.join(BASE, "output", "web", "basemap.json")
DEST = os.path.join(os.path.dirname(BASE), os.pardir, "hri-explorer-site", "index.html")

payload = json.load(open(DATA))
meta = payload["meta"]
bm = json.load(open(BASEMAP)) if os.path.exists(BASEMAP) else {
    "coast": [], "water": [], "park": [], "edges": []}

# Set this to the key from https://carto.com/basemaps/apikey/ and re-run.
# Left as the placeholder, the CARTO options are disabled in the basemap picker
# so the page can never ship watermarked tiles by accident.
CARTO_KEY = os.environ.get("CARTO_KEY", "YOUR_CARTO_API_KEY")

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perth Hub Reliance Index</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
  integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<style>
  :root{ color-scheme: light;
    --ink:#0F1E3D; --muted:#64748B; --line:#E2E8F0; --bg:#FBFCFD;
    --panel:#FFFFFF; --accent:#0E7C86; --warm:#C2410C; }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    color:var(--ink); background:#fff}
  #hri-app{display:flex; flex-direction:column; height:100vh}
  .hd{padding:11px 16px; border-bottom:1px solid var(--line); background:var(--panel);
    display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; flex:none}
  .hd h2{margin:0; font-size:15px; font-weight:650; letter-spacing:-.01em}
  .hd .sub{font-size:11.5px; color:var(--muted)}
  .body{flex:1; display:flex; min-height:0}
  #map{flex:1; min-width:0; background:#EAEFF3}
  .side{width:312px; border-left:1px solid var(--line); background:var(--panel);
    display:flex; flex-direction:column; min-height:0}
  .ctl{padding:10px 13px; border-bottom:1px solid var(--line)}
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
  .item .rk{font-variant-numeric:tabular-nums; color:var(--muted); font-size:10.5px; width:24px; flex:none}
  .item .nm{flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
  .item .sb{font-size:10px; color:var(--muted)}
  .item .vl{font-variant-numeric:tabular-nums; font-weight:650; color:var(--accent); font-size:11.5px}
  .det{border-top:1px solid var(--line); padding:11px 13px; background:#FCFDFE;
    max-height:250px; overflow-y:auto}
  .det h3{margin:0 0 1px; font-size:13px; font-weight:650}
  .det .sb2{font-size:11px; color:var(--muted); margin-bottom:8px}
  .kv{display:grid; grid-template-columns:1fr auto; gap:2px 10px; font-size:11.5px}
  .kv .k{color:var(--muted)}
  .kv .v{font-variant-numeric:tabular-nums; font-weight:600}
  .bar{height:3px; background:var(--line); border-radius:2px; overflow:hidden;
    grid-column:1/-1; margin:1px 0 4px}
  .bar i{display:block; height:100%; background:var(--accent)}
  .tag{display:inline-block; padding:1px 7px; border-radius:20px; font-size:9.5px;
    font-weight:650; text-transform:uppercase; letter-spacing:.04em; color:#fff}
  .lg{position:absolute; left:12px; bottom:22px; z-index:500;
    background:rgba(255,255,255,.94); border:1px solid var(--line);
    border-radius:7px; padding:8px 10px; font-size:10.5px}
  .lg .ramp{height:8px; width:150px; border-radius:2px; margin:4px 0 3px}
  .lg .ends{display:flex; justify-content:space-between; color:var(--muted);
    font-variant-numeric:tabular-nums}
  #ov{position:absolute; top:0; left:0; pointer-events:none; z-index:400}
  .leaflet-container{font-family:inherit}
  @media (max-width:820px){ .body{flex-direction:column} .side{width:auto;
    border-left:none; border-top:1px solid var(--line); max-height:48vh} }
</style>
</head>
<body>
<div id="hri-app">
  <div class="hd">
    <h2>Perth Hub Reliance Index</h2>
    <span class="sub" id="metaline"></span>
  </div>
  <div class="body">
    <div id="map">
      <div class="lg">
        <div id="lgTitle" style="font-weight:650"></div>
        <div class="ramp" id="lgRamp"></div>
        <div class="ends"><span id="lgLo"></span><span id="lgHi"></span></div>
      </div>
    </div>
    <div class="side">
      <div class="ctl"><label>Basemap</label><select id="baseSel"></select></div>
      <div class="ctl"><label>Colour by</label><select id="varSel"></select></div>
      <div class="ctl">
        <label for="beta">&beta; &mdash; reliance amplification</label>
        <div class="brow"><span>0</span><span class="bval" id="bOut">0.50</span><span>1.5</span></div>
        <input type="range" id="beta" min="0" max="1.5" step="0.05" value="0.5">
        <div class="brow"><span id="bNote"></span></div>
      </div>
      <div class="ctl"><label>Typology</label><div class="chips" id="chips"></div></div>
      <div class="ctl"><label>Layers</label><div class="chips" id="layers"></div></div>
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
const EDG = BM.edges || [];
const CARTO_KEY = "__CARTO_KEY__";
const HAS_CARTO = CARTO_KEY && CARTO_KEY !== "YOUR_" + "CARTO_API_KEY";
const N = D.meta.n;

let beta = D.meta.beta_default;
const hri = new Float32Array(N);
const rank = new Int32Array(N);
let order = null, sel = -1;

const TYPES = {
  critical:  {c:'#0F1E3D', label:'critical'},
  captive:   {c:'#C2410C', label:'captive'},
  integrated:{c:'#0E7C86', label:'integrated'},
  redundant: {c:'#94A3B8', label:'redundant'}
};
function median(a){const b=Float64Array.from(a);b.sort();return b[b.length>>1];}
const medS = median(D.s), medL = median(D.l);
const typ = new Array(N);
for(let i=0;i<N;i++){
  const hS = D.s[i] > medS, hL = D.l[i] > medL;
  typ[i] = hS ? (hL?'critical':'integrated') : (hL?'captive':'redundant');
}
const active = new Set(Object.keys(TYPES));
const layers = {network:false};

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

const RAMP = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function ramp(t){
  t=Math.max(0,Math.min(1,t));
  const x=t*(RAMP.length-1), i=Math.floor(x), f=x-i;
  const a=RAMP[i], b=RAMP[Math.min(i+1,RAMP.length-1)];
  return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+Math.round(a[1]+(b[1]-a[1])*f)
        +','+Math.round(a[2]+(b[2]-a[2])*f)+')';
}
let breaks=[];
function computeBreaks(){
  if(cur.cat){breaks=[];return;}
  const v=[]; for(let i=0;i<N;i++) v.push(cur.get(i));
  v.sort((a,b)=>a-b);
  breaks=[]; for(let q=0;q<=10;q++) breaks.push(v[Math.min(v.length-1,Math.floor(q/10*(v.length-1)))]);
}
function colOf(i){
  if(cur.cat) return TYPES[typ[i]].c;
  const v=cur.get(i); let p=1;
  for(let q=1;q<breaks.length;q++){
    if(v<=breaks[q]){
      p=(q-1+(breaks[q]>breaks[q-1]?(v-breaks[q-1])/(breaks[q]-breaks[q-1]):0))/10;
      break;
    }
  }
  return ramp(p);
}

// ───────────────────────────────────────────────────────── map + tiles
// zoomAnimation is off so the canvas overlay cannot drift mid-zoom; zooming
// becomes instant rather than eased, which is the right trade for 11,773 points.
const map = L.map('map', {preferCanvas:true, zoomAnimation:false,
                         zoomControl:true, minZoom:8, maxZoom:18});

const OSM_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
const CARTO_ATTR = OSM_ATTR + ' &copy; <a href="https://carto.com/attribution/">CARTO</a>';
// CARTO raster basemaps have required an API key since 23 September 2026;
// without one the tiles are served watermarked. The key is appended as a query
// parameter per CARTO's documentation.
function carto(style){
  return L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/'+style+'/{z}/{x}/{y}{r}.png?key='+CARTO_KEY,
    {attribution: CARTO_ATTR, subdomains:'abcd', maxZoom:19});
}
const BASES = [
  {k:'osm',    n:'OpenStreetMap',        layer:()=>L.tileLayer(
     'https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:OSM_ATTR,maxZoom:19})},
  {k:'simple', n:'Simple \u2014 no tiles', layer:null},
  {k:'positron', n:'CARTO Positron',      layer:()=>carto('light_all'), carto:true},
  {k:'nolabels', n:'CARTO, no labels',    layer:()=>carto('light_nolabels'), carto:true}
];
let baseMode='osm', tileLayer=null;
const attribCtl = map.attributionControl;
function setBase(k){
  baseMode=k;
  if(tileLayer){ map.removeLayer(tileLayer); tileLayer=null; }
  const b=BASES.find(x=>x.k===k);
  if(b && b.layer){ tileLayer=b.layer(); tileLayer.addTo(map); }
  const c=map.getContainer();
  c.style.background = (k==='simple') ? '#E8EFF5' : '#EAEFF3';
  if(k==='simple') attribCtl.addAttribution(SIMPLE_ATTR);
  else attribCtl.removeAttribution(SIMPLE_ATTR);
  draw();
}
const SIMPLE_ATTR = 'Coast &amp; land use: '
  + '<a href="https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3">ABS ASGS 2021</a>';

let minLat=90,maxLat=-90,minLon=180,maxLon=-180;
for(let i=0;i<N;i++){
  if(D.lat[i]<minLat)minLat=D.lat[i]; if(D.lat[i]>maxLat)maxLat=D.lat[i];
  if(D.lon[i]<minLon)minLon=D.lon[i]; if(D.lon[i]>maxLon)maxLon=D.lon[i];
}
map.fitBounds([[minLat,minLon],[maxLat,maxLon]], {padding:[12,12]});

// ─────────────────────────────────────────────── canvas overlay for stops
const cv = document.createElement('canvas');
cv.id='ov';
map.getContainer().appendChild(cv);
const ctx = cv.getContext('2d');
let px=new Float64Array(N), py=new Float64Array(N), pz=-1, dpr=1;
let RINGS={coast:[],water:[],park:[]}, rz=-1;

function project(){
  const z=map.getZoom();
  if(z===pz) return;
  pz=z;
  for(let i=0;i<N;i++){
    const p=map.project([D.lat[i],D.lon[i]], z);
    px[i]=p.x; py[i]=p.y;
  }
}
function projectRings(){
  const z=map.getZoom();
  if(z===rz) return;
  rz=z;
  ['coast','water','park'].forEach(k=>{
    RINGS[k]=(BM[k]||[]).map(r=>{
      const a=new Float64Array(r.length*2);
      for(let j=0;j<r.length;j++){
        const p=map.project([r[j][1],r[j][0]], z);
        a[j*2]=p.x; a[j*2+1]=p.y;
      }
      return a;
    });
  });
}
function drawVectorBase(o,s){
  projectRings();
  const path=fr=>{
    ctx.beginPath();
    for(let j=0;j<fr.length;j+=2){
      const X=fr[j]-o.x, Y=fr[j+1]-o.y;
      if(j===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y);
    }
    ctx.closePath();
  };
  ctx.fillStyle='#FAFBFC'; ctx.strokeStyle='#DCE3EA'; ctx.lineWidth=1;
  RINGS.coast.forEach(fr=>{ path(fr); ctx.fill(); ctx.stroke(); });
  ctx.fillStyle='#EDF3EC'; RINGS.park.forEach(fr=>{ path(fr); ctx.fill(); });
  ctx.fillStyle='#D6E6F2'; RINGS.water.forEach(fr=>{ path(fr); ctx.fill(); });
}
function sizeCanvas(){
  const s=map.getSize(); dpr=window.devicePixelRatio||1;
  cv.width=s.x*dpr; cv.height=s.y*dpr;
  cv.style.width=s.x+'px'; cv.style.height=s.y+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
function draw(){
  project();
  const o=map.getPixelBounds().min, s=map.getSize();
  ctx.clearRect(0,0,s.x,s.y);
  if(baseMode==='simple') drawVectorBase(o,s);
  const z=map.getZoom(), r=Math.max(1.3, (z-8)*0.72);

  if(layers.network && EDG.length){
    ctx.strokeStyle='#7C8CA0'; ctx.lineWidth=Math.max(0.35,(z-9)*0.18);
    ctx.globalAlpha=0.45; ctx.beginPath();
    for(let e=0;e<EDG.length;e+=2){
      const a=EDG[e], b=EDG[e+1];
      const ax=px[a]-o.x, ay=py[a]-o.y;
      if(ax<-80||ax>s.x+80||ay<-80||ay>s.y+80) continue;
      ctx.moveTo(ax,ay); ctx.lineTo(px[b]-o.x, py[b]-o.y);
    }
    ctx.stroke(); ctx.globalAlpha=1;
  }

  for(let pass=0;pass<2;pass++){
    for(let i=0;i<N;i++){
      const on=active.has(typ[i]); if(pass===0? on : !on) continue;
      const X=px[i]-o.x, Y=py[i]-o.y;
      if(X<-6||X>s.x+6||Y<-6||Y>s.y+6) continue;
      ctx.globalAlpha = on?0.88:0.06;
      ctx.fillStyle = on?colOf(i):'#94A3B8';
      ctx.beginPath(); ctx.arc(X,Y,r,0,6.283); ctx.fill();
    }
  }
  ctx.globalAlpha=1;
  if(sel>=0){
    const X=px[sel]-o.x, Y=py[sel]-o.y, rr=Math.max(8,r+5);
    ctx.strokeStyle='#0F1E3D'; ctx.lineWidth=2.5;
    ctx.beginPath(); ctx.arc(X,Y,rr,0,6.283); ctx.stroke();
    ctx.strokeStyle='#fff'; ctx.lineWidth=1.2;
    ctx.beginPath(); ctx.arc(X,Y,rr+1.8,0,6.283); ctx.stroke();
  }
}
map.on('move zoom moveend zoomend viewreset', draw);
map.on('resize', ()=>{ sizeCanvas(); draw(); });
map.on('zoomend', ()=>{ rz=-1; });

map.on('click', e=>{
  const o=map.getPixelBounds().min;
  const cx=e.containerPoint.x, cy=e.containerPoint.y;
  let best=-1, bd=400;
  for(let i=0;i<N;i++){
    if(!active.has(typ[i])) continue;
    const dx=(px[i]-o.x)-cx, dy=(py[i]-o.y)-cy, dd=dx*dx+dy*dy;
    if(dd<bd){bd=dd;best=i;}
  }
  if(best>=0){ sel=best; showDetail(best); draw(); }
});

// ─────────────────────────────────────────────────────────── recompute
function recompute(){
  for(let i=0;i<N;i++) hri[i]=D.s[i]*(1+beta*D.l[i]);
  order=Array.from({length:N},(_,i)=>i).sort((a,b)=>hri[b]-hri[a]);
  for(let k=0;k<N;k++) rank[order[k]]=k+1;
  computeBreaks(); renderList(); updateLegend();
  if(sel>=0) showDetail(sel);
  draw();
}

const listEl=document.getElementById('list'), detEl=document.getElementById('det');
const qEl=document.getElementById('q');
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function renderList(){
  const q=qEl.value.trim().toLowerCase(); const rows=[];
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
    const i=+el.dataset.i; sel=i;
    map.setView([D.lat[i],D.lon[i]], Math.max(map.getZoom(),15));
    showDetail(i); draw();
  });
}
function showDetail(i){
  const t=TYPES[typ[i]], pct=v=>Math.max(0,Math.min(1,v))*100;
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
   +'<span class="k">G peak delay</span><span class="v">'+(D.g[i]?('+'+D.g[i].toFixed(0)+' s'):'—')+'</span>'
   +'<span class="k">alternatives nearby (A)</span><span class="v">'+D.alt[i]+'</span>'
   +'<span class="k">departures AM / off / PM / rem</span><span class="v">'
     +D.am[i]+' / '+D.off[i]+' / '+D.pm[i]+' / '+D.rem[i]+'</span>'
   +'</div>';
}
function fmt(v,d){return d===0?Math.round(v).toLocaleString():v.toFixed(d);}
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

const baseSel=document.getElementById('baseSel');
BASES.forEach(b=>{
  const o=document.createElement('option');
  o.value=b.k;
  o.textContent=b.n + (b.carto && !HAS_CARTO ? '  \u2014 needs API key' : '');
  if(b.carto && !HAS_CARTO) o.disabled=true;
  baseSel.appendChild(o);
});
baseSel.value='osm';
baseSel.onchange=()=>setBase(baseSel.value);

const sel1=document.getElementById('varSel');
VARS.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v.n;sel1.appendChild(o);});
sel1.onchange=()=>{cur=VARS[+sel1.value];computeBreaks();updateLegend();draw();};

const chips=document.getElementById('chips');
Object.entries(TYPES).forEach(([k,t])=>{
  const c=document.createElement('div'); c.className='chip';
  c.innerHTML='<span class="dot" style="background:'+t.c+'"></span>'+t.label;
  c.onclick=()=>{ active.has(k)?active.delete(k):active.add(k);
    c.classList.toggle('off',!active.has(k)); renderList(); draw(); };
  chips.appendChild(c);
});
const layEl=document.getElementById('layers');
[['network','Network edges']].forEach(([k,label])=>{
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
    : (beta<0.35?'below the stability band'
      :(beta<=1.0?'inside the stability band [0.35, 1.00]':'above the stability band'));
  recompute();
};
qEl.oninput=renderList;

document.getElementById('metaline').textContent =
  N.toLocaleString()+' merged nodes · CRITIC weights ('+D.meta.weight_frame
  +' frame) · T at '+D.meta.t_threshold+' · HRI = S₊ × (1 + β · L)';

sizeCanvas();
setBase('osm');
bEl.value=beta; bOut.textContent=beta.toFixed(2); bEl.oninput();
setTimeout(()=>{ map.invalidateSize(); sizeCanvas(); pz=-1; draw(); }, 60);
})();
</script>
</body></html>
"""

out = (HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
           .replace("__BASEMAP__", json.dumps(bm, separators=(",", ":")))
           .replace("__CARTO_KEY__", CARTO_KEY))
os.makedirs(os.path.dirname(DEST), exist_ok=True)
open(DEST, "w").write(out)
print(f"wrote {DEST}  {os.path.getsize(DEST)/1e6:.2f} MB")
print(f"  {meta['n']:,} nodes, {len(bm.get('edges', []))//2:,} edges, "
      f"beta default {meta['beta_default']}, T {meta['t_threshold']}")
print("  CARTO key: " + ("SET" if CARTO_KEY != "YOUR_CARTO_API_KEY"
                         else "placeholder (CARTO options disabled)"))
