// Replicates app.js refit()/margin()/winProb() exactly, outside the DOM.
const F = require('/Users/benrosen/Documents/march-madness-forecaster/docs/fit.js');
const fs = require('fs');
const R = '/Users/benrosen/Documents/march-madness-forecaster/docs/data/';
const CANON = ['barthag','t_rank','sos_avg_opp_barthag','adj_offensive_efficiency','adj_defensive_efficiency','adj_tempo','effective_fg_pct','three_pt_pct','three_pt_rate','offensive_reb_rate','turnover_rate'];
const year = +process.argv[2];
const tr = JSON.parse(fs.readFileSync(R+'training.json'));
const season = JSON.parse(fs.readFileSync(R+`season_${year}.json`));
const cols=[], keys=[];
for (const k of CANON){ const i=tr.keys.indexOf(k); if(i>=0){keys.push(k);cols.push(i);} }
const f = F.fitLinear(tr.games, cols, year);
const oos = F.crossValidate(tr.games, cols, tr.years, 2014);      // <- app.js: ALL years >= 2014
const calUI = oos.calibration;
// causal alternative: same as model_baseline.js / pit_production_model.py
const oosPrior = F.crossValidate(tr.games.filter(g=>g.y<year), cols, tr.years.filter(y=>y<year), 2014);
let calCausal = {a:1, nu:Infinity};
if (oosPrior) { const n=oosPrior.n, w=n/(n+63); calCausal = {a: w*oosPrior.calibration.a + (1-w), nu: oosPrior.calibration.nu}; }
const z = season.z, teams = season.teams;
const diff=(a,b)=>keys.map(k=>{const c=z[k]; return c?((c[a]||0)-(c[b]||0)):0;});
const margin=(a,b)=>{const x=diff(a,b); let t=0; for(let j=0;j<keys.length;j++) t+=f.beta[j]*x[j]; return t;};
const rows=[];
const fr = season.first_round;
for (let g=0; g<fr.length; g+=2){ const a=fr[g], b=fr[g+1];
  rows.push({a: teams[a].id, b: teams[b].id, sa: teams[a].seed, sb: teams[b].seed, m: margin(a,b),
    p_ui: F.winProbFromMargin(margin(a,b), f.sigma, calUI), p_causal: F.winProbFromMargin(margin(a,b), f.sigma, calCausal),
    p_ui_rev: F.winProbFromMargin(margin(b,a), f.sigma, calUI)}); }
console.log(JSON.stringify({year, sigma:f.sigma, beta:f.beta, keys, calUI, calCausal, oosN: oos.n, oosPriorN: oosPrior?oosPrior.n:0, rows}));
