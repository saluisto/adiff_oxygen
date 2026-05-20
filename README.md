[oxygen_static.html](https://github.com/user-attachments/files/28046437/oxygen_static.html)# adiff_oxygen

Dynamic oxygen model for a well-mixed pond using automatic differentiation in TensorFlow. 

## Mathematical model:

[Uploading oxygen_stati<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Oxygen Mass Balance Model — Static Structure</title>
<style>
  :root {
    --bg:            #0a0e1a;
    --bg-card:       #131826;
    --text:          #e6edf3;
    --text-dim:      #9aa5b1;
    --border-dim:    #1f2937;

    /* Neon palette — used for card outlines and equation-box accents */
    --neon-yellow:   #ffe45e;
    --neon-blue:     #38bdf8;
    --neon-green:    #4ade80;
    --neon-orange:   #fb923c;
    --neon-grey:     #a3a3a3;
    --neon-pink:     #f472b6;
    --neon-red:      #f87171;
    --neon-purple:   #c084fc;

    /* Parameter-type color coding — matches the equation-tree legend */
    --col-state:     #4ade80;
    --col-constant:  #38bdf8;
    --col-aux:       #f87171;
    --col-tab:       #c084fc;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: radial-gradient(ellipse at top, #11182a 0%, var(--bg) 60%);
    color: var(--text);
    min-height: 100vh;
    padding: 28px 20px 80px;
  }
  h1 {
    margin: 0 0 6px 0;
    font-size: 26px;
    letter-spacing: 0.5px;
    text-align: center;
    color: #fff;
    text-shadow: 0 0 12px rgba(255, 228, 94, 0.35);
  }
  .subtitle {
    text-align: center;
    color: var(--text-dim);
    margin-bottom: 28px;
    font-size: 14px;
  }

  /* ===== Equation rendering (inline HTML, no MathJax) ===== */
  .eq {
    font-family: 'Cambria Math', 'Latin Modern Math', 'STIX Two Math', Georgia, serif;
    font-size: 16px;
    line-height: 2.2;
    color: var(--text);
  }
  .eq.big { font-size: 20px; }
  .eq .v { font-style: italic; }
  .eq sub, .eq sup {
    font-size: 0.72em;
    line-height: 0;
    position: relative;
  }
  .eq sub { top: 0.35em; }
  .eq sup { top: -0.55em; }

  .eq .frac {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    vertical-align: middle;
    text-align: center;
    margin: 0 0.2em;
    font-size: 0.9em;
    line-height: 1.05;
    white-space: nowrap;
  }
  .eq .frac > .num-f {
    border-bottom: 1.5px solid currentColor;
    padding: 0.05em 0.35em 0.3em;
    display: block;
  }
  .eq .frac > .den-f {
    padding: 0.3em 0.35em 0.05em;
    display: block;
  }
  .eq .frac sub { top: 0.15em; }
  .eq .frac sup { top: -0.4em; }

  .eq .unit {
    display: inline-block;
    margin-left: 10px;
    padding: 2px 10px;
    border-radius: 6px;
    background: rgba(255,255,255,0.06);
    font-size: 12px;
    color: var(--text-dim);
    font-family: 'Inter', sans-serif;
    font-style: normal;
    vertical-align: middle;
  }

  /* ===== Tree layout ===== */
  .tree-wrap {
    max-width: 1500px;
    margin: 0 auto;
    overflow-x: auto;
  }

  .level {
    display: flex;
    justify-content: center;
    gap: 28px;
    flex-wrap: wrap;
    margin-bottom: 14px;
    position: relative;
  }

  /* Connector below a node when its children are expanded */
  .has-children.expanded > .node::after {
    content: "";
    display: block;
    width: 2px;
    height: 18px;
    background: linear-gradient(to bottom, currentColor, transparent);
    margin: 4px auto -2px;
    opacity: 0.55;
  }

  /* ===== Equation node ===== */
  .node {
    position: relative;
    padding: 14px 18px;
    border-radius: 14px;
    background: var(--bg-card);
    border: 2px solid;
    min-width: 240px;
    user-select: none;
  }
  .node .eqnum {
    position: absolute;
    top: -10px;
    left: -10px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--bg);
    border: 2px solid currentColor;
    font-size: 13px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    color: currentColor;
  }
  .node .label {
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }

  /* Color variants per node */
  .c-yellow { color: var(--neon-yellow); box-shadow: 0 0 18px rgba(255, 228, 94, 0.18); }
  .c-blue   { color: var(--neon-blue);   box-shadow: 0 0 18px rgba(56, 189, 248, 0.18); }
  .c-green  { color: var(--neon-green);  box-shadow: 0 0 18px rgba(74, 222, 128, 0.18); }
  .c-orange { color: var(--neon-orange); box-shadow: 0 0 18px rgba(251, 146, 60, 0.18); }
  .c-grey   { color: var(--neon-grey);   box-shadow: 0 0 18px rgba(163, 163, 163, 0.15); }
  .c-pink   { color: var(--neon-pink);   box-shadow: 0 0 18px rgba(244, 114, 182, 0.18); }

  /* The child container */
  .children {
    display: none;
    width: 100%;
    margin-top: 18px;
    padding-top: 6px;
    border-top: 1px dashed var(--border-dim);
  }
  .has-children.expanded > .children,
  .leaf.expanded > .children {
    display: block;
  }

  /* Make horizontal levels arrange children side by side */
  .children > .level {
    margin-bottom: 0;
  }

  /* Initial-condition note */
  .ic-note {
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 6px;
    font-family: 'Inter', sans-serif;
  }

  /* Parameter detail panel inside an expanded node */
  .params {
    margin-top: 10px;
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--border-dim);
    font-size: 12.5px;
  }
  .params-title {
    color: var(--text-dim);
    font-size: 11px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .param-row {
    display: grid;
    grid-template-columns: minmax(60px, auto) 1fr auto;
    gap: 10px;
    padding: 3px 0;
    border-bottom: 1px dotted rgba(255,255,255,0.05);
  }
  .param-row:last-child { border-bottom: none; }
  .param-sym {
    font-weight: 600;
    font-style: italic;
  }
  .param-desc { color: var(--text-dim); }
  .param-unit { color: var(--text-dim); font-size: 11px; }

  /* Param type tag at the start of symbol */
  .pt-state    { color: var(--col-state); }
  .pt-constant { color: var(--col-constant); }
  .pt-aux      { color: var(--col-aux); }
  .pt-tab      { color: var(--col-tab); }

  /* Legend */
  .legend {
    max-width: 1100px;
    margin: 0 auto 28px;
    display: flex;
    flex-wrap: wrap;
    gap: 14px 24px;
    justify-content: center;
    padding: 12px 18px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
  }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-dim); }
  .legend-dot {
    width: 12px; height: 12px; border-radius: 50%;
    box-shadow: 0 0 8px currentColor;
  }

  @media (max-width: 800px) {
    .level { flex-direction: column; align-items: stretch; }
  }
</style>
</head>
<body>

<h1>Oxygen Mass Balance Model</h1>
<div class="subtitle">Full structure — all levels expanded</div>

<!-- Legend for parameter color codes -->
<div class="legend">
  <div class="legend-item"><span class="legend-dot" style="background: var(--col-state); color: var(--col-state);"></span> State variable</div>
  <div class="legend-item"><span class="legend-dot" style="background: var(--col-constant); color: var(--col-constant);"></span> Constant</div>
  <div class="legend-item"><span class="legend-dot" style="background: var(--col-aux); color: var(--col-aux);"></span> Auxiliary / time-dependent</div>
  <div class="legend-item"><span class="legend-dot" style="background: var(--col-tab); color: var(--col-tab);"></span> Tabular / measurement</div>
</div>

<div class="tree-wrap">

  <!-- ROOT: Equation 1 - Oxygen Mass Balance -->
  <div class="level">
    <div class="has-children expanded c-yellow" id="n1">
      <div class="node">
        <span class="eqnum">1</span>
        <div class="label">Oxygen Mass Balance Model</div>
        <div class="eq big">
          <span class="frac"><span class="num-f">d<span class="v">O</span><sub>2</sub></span><span class="den-f">d<span class="v">t</span></span></span>
          = <span class="v">F</span><sub>atm</sub>(<span class="v">t</span>)
          + <span class="v">F</span><sub>prim</sub>(<span class="v">t</span>)
          + <span class="v">F</span><sub>mac</sub>(<span class="v">t</span>)
          − <span class="v">F</span><sub>SOD</sub>
          <span class="unit">mg L⁻¹ min⁻¹</span>
        </div>
        <div class="ic-note">Initial condition: O₂(t=0) = measured O₂</div>
      </div>

      <div class="children">
        <div class="params">
          <div class="params-title">Parameters of Eq. 1</div>
          <div class="param-row"><span class="param-sym pt-state">O₂</span><span class="param-desc">Oxygen concentration</span><span class="param-unit">mg L⁻¹</span></div>
          <div class="param-row"><span class="param-sym pt-aux">t</span><span class="param-desc">Time</span><span class="param-unit">min</span></div>
          <div class="param-row"><span class="param-sym pt-aux">F<sub>atm</sub>(t)</span><span class="param-desc">Exchange flux with atmosphere</span><span class="param-unit">mg L⁻¹ min⁻¹</span></div>
          <div class="param-row"><span class="param-sym pt-aux">F<sub>prim</sub>(t)</span><span class="param-desc">Primary production</span><span class="param-unit">mg L⁻¹ min⁻¹</span></div>
          <div class="param-row"><span class="param-sym pt-aux">F<sub>mac</sub>(t)</span><span class="param-desc">Macrophytes oxygen production</span><span class="param-unit">mg L⁻¹ min⁻¹</span></div>
          <div class="param-row"><span class="param-sym pt-aux">F<sub>SOD</sub></span><span class="param-desc">Sediment oxygen demand</span><span class="param-unit">mg L⁻¹ min⁻¹</span></div>
        </div>

        <!-- Sub-experiments level -->
        <div class="level">
          <!-- Reaeration Experiment - Eq 3 -->
          <div class="has-children expanded c-blue" id="n3">
            <div class="node">
              <span class="eqnum">3</span>
              <div class="label">Reaeration Experiment</div>
              <div class="eq">
                <span class="v">F</span><sub>atm</sub>(<span class="v">t</span>) =
                <span class="frac"><span class="num-f"><span class="v">K</span><sub>l</sub></span><span class="den-f"><span class="v">H</span></span></span>
                <span style="font-size:1.2em;">(</span><span class="v">C</span><sub>s</sub>(<span class="v">t</span>) − <span class="v">O</span><sub>2</sub>(<span class="v">t</span>)<span style="font-size:1.2em;">)</span>
                <span class="unit">mg L⁻¹ min⁻¹</span>
              </div>
            </div>
            <div class="children">
              <div class="params">
                <div class="params-title">Parameters of Eq. 3</div>
                <div class="param-row"><span class="param-sym pt-aux">F<sub>atm</sub>(t)</span><span class="param-desc">Exchange flux with atmosphere</span><span class="param-unit">mg L⁻¹ min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-constant">K<sub>l</sub></span><span class="param-desc">Mass transfer coefficient</span><span class="param-unit">m min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-constant">H</span><span class="param-desc">Average depth of the pond</span><span class="param-unit">m</span></div>
                <div class="param-row"><span class="param-sym pt-aux">C<sub>s</sub>(t)</span><span class="param-desc">Oxygen saturation</span><span class="param-unit">mg L⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-state">O₂(t)</span><span class="param-desc">Oxygen concentration</span><span class="param-unit">mg L⁻¹</span></div>
              </div>
              <!-- Eq 7: oxygen saturation -->
              <div class="level">
                <div class="leaf expanded c-blue" id="n7">
                  <div class="node">
                    <span class="eqnum">7</span>
                    <div class="label">Oxygen Saturation</div>
                    <div class="eq">
                      <span class="v">C</span><sub>s</sub>(<span class="v">t</span>) = 14.652 − 0.41022 <span class="v">T</span>(<span class="v">t</span>) + 0.007991 <span class="v">T</span>(<span class="v">t</span>)<sup>2</sup> − 0.000077774 <span class="v">T</span>(<span class="v">t</span>)<sup>3</sup>
                      <span class="unit">mg L⁻¹</span>
                    </div>
                  </div>
                  <div class="children">
                    <div class="params">
                      <div class="params-title">Parameters of Eq. 7</div>
                      <div class="param-row"><span class="param-sym pt-aux">C<sub>s</sub>(t)</span><span class="param-desc">Oxygen saturation</span><span class="param-unit">mg L⁻¹</span></div>
                      <div class="param-row"><span class="param-sym pt-tab">T(t)</span><span class="param-desc">Temperature (measured)</span><span class="param-unit">°C</span></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Algae Experiment - Eq 4 -->
          <div class="has-children expanded c-green" id="n4">
            <div class="node">
              <span class="eqnum">4</span>
              <div class="label">Algae Experiment</div>
              <div class="eq">
                <span class="v">F</span><sub>prim</sub>(<span class="v">t</span>) =
                <span class="frac"><span class="num-f">d<span class="v">A</span></span><span class="den-f">d<span class="v">t</span></span></span>
                · <span class="v">c</span> =
                <span style="font-size:1.2em;">(</span>μ(<span class="v">t</span>) − <span class="v">R</span><span style="font-size:1.2em;">)</span>
                <span class="v">A</span>(<span class="v">t</span>) <span class="v">c</span>
                <span class="unit">mg L⁻¹ min⁻¹</span>
              </div>
            </div>
            <div class="children">
              <div class="params">
                <div class="params-title">Parameters of Eq. 4</div>
                <div class="param-row"><span class="param-sym pt-aux">F<sub>prim</sub>(t)</span><span class="param-desc">Primary production</span><span class="param-unit">mg L⁻¹ min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-state">A(t)</span><span class="param-desc">Algal biomass</span><span class="param-unit">mg L⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-aux">t</span><span class="param-desc">Time</span><span class="param-unit">min</span></div>
                <div class="param-row"><span class="param-sym pt-aux">μ(t)</span><span class="param-desc">Algal growth rate</span><span class="param-unit">min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-constant">R</span><span class="param-desc">Algal respiration rate</span><span class="param-unit">min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-constant">c</span><span class="param-desc">O₂ produced per g algal production</span><span class="param-unit">– (≈2)</span></div>
              </div>

              <!-- Children of Eq 4: Algae Growth Model (Eq 2) and Growth rate μ (Eq 8) -->
              <div class="level">
                <!-- Eq 2 -->
                <div class="leaf expanded c-green" id="n2">
                  <div class="node">
                    <span class="eqnum">2</span>
                    <div class="label">Algae Growth Model</div>
                    <div class="eq">
                      <span class="frac"><span class="num-f">d<span class="v">A</span></span><span class="den-f">d<span class="v">t</span></span></span>
                      = <span style="font-size:1.2em;">(</span>μ(<span class="v">t</span>) − <span class="v">R</span><span style="font-size:1.2em;">)</span>
                      <span class="v">A</span>(<span class="v">t</span>)
                      <span class="unit">mg L⁻¹ min⁻¹</span>
                    </div>
                    <div class="ic-note">Initial condition: A(t=0) = 0.6 mg/L</div>
                  </div>
                  <div class="children">
                    <div class="params">
                      <div class="params-title">Parameters of Eq. 2</div>
                      <div class="param-row"><span class="param-sym pt-state">A(t)</span><span class="param-desc">Algal biomass</span><span class="param-unit">mg L⁻¹</span></div>
                      <div class="param-row"><span class="param-sym pt-aux">t</span><span class="param-desc">Time</span><span class="param-unit">min</span></div>
                      <div class="param-row"><span class="param-sym pt-aux">μ(t)</span><span class="param-desc">Algal growth rate</span><span class="param-unit">min⁻¹</span></div>
                      <div class="param-row"><span class="param-sym pt-constant">R</span><span class="param-desc">Algal respiration rate</span><span class="param-unit">min⁻¹</span></div>
                    </div>
                  </div>
                </div>

                <!-- Eq 8: μ -->
                <div class="has-children expanded c-green" id="n8">
                  <div class="node">
                    <span class="eqnum">8</span>
                    <div class="label">Algal Growth Rate</div>
                    <div class="eq">
                      μ(<span class="v">t</span>) = μ<sub>max</sub> <span class="v">F</span><sub>N</sub> <span class="v">F</span><sub>L</sub>(<span class="v">t</span>) <span class="v">F</span><sub>T1</sub>(<span class="v">t</span>)
                      <span class="unit">min⁻¹</span>
                    </div>
                  </div>
                  <div class="children">
                    <div class="params">
                      <div class="params-title">Parameters of Eq. 8</div>
                      <div class="param-row"><span class="param-sym pt-aux">μ(t)</span><span class="param-desc">Algal growth rate</span><span class="param-unit">min⁻¹</span></div>
                      <div class="param-row"><span class="param-sym pt-constant">μ<sub>max</sub></span><span class="param-desc">Max growth rate</span><span class="param-unit">min⁻¹</span></div>
                      <div class="param-row"><span class="param-sym pt-constant">F<sub>N</sub></span><span class="param-desc">Nutrient limitation</span><span class="param-unit">– (= 1)</span></div>
                      <div class="param-row"><span class="param-sym pt-aux">F<sub>L</sub>(t)</span><span class="param-desc">Light limitation</span><span class="param-unit">–</span></div>
                      <div class="param-row"><span class="param-sym pt-aux">F<sub>T1</sub>(t)</span><span class="param-desc">Temperature limitation</span><span class="param-unit">–</span></div>
                    </div>

                    <!-- Children of Eq 8: F_T1 (Eq 10) and F_L (Eq 12) -->
                    <div class="level">
                      <!-- Eq 10 -->
                      <div class="leaf expanded c-green" id="n10">
                        <div class="node">
                          <span class="eqnum">10</span>
                          <div class="label">Temperature Limitation (algae)</div>
                          <div class="eq">
                            <span class="v">F</span><sub>T1</sub>(<span class="v">t</span>) = Θ<sub>1</sub><sup>(<span class="v">T</span>(<span class="v">t</span>) − 20°C) / γ</sup>
                            <span class="unit">–</span>
                          </div>
                        </div>
                        <div class="children">
                          <div class="params">
                            <div class="params-title">Parameters of Eq. 10</div>
                            <div class="param-row"><span class="param-sym pt-aux">F<sub>T1</sub>(t)</span><span class="param-desc">Temperature limitation</span><span class="param-unit">–</span></div>
                            <div class="param-row"><span class="param-sym pt-constant">Θ₁</span><span class="param-desc">Temperature coefficient algae (1.02–1.06)</span><span class="param-unit">– (≈1.02)</span></div>
                            <div class="param-row"><span class="param-sym pt-tab">T(t)</span><span class="param-desc">Temperature (measured)</span><span class="param-unit">°C</span></div>
                            <div class="param-row"><span class="param-sym pt-constant">γ</span><span class="param-desc">Scaling parameter</span><span class="param-unit">°C (= 1)</span></div>
                          </div>
                        </div>
                      </div>

                      <!-- Eq 12: F_L -->
                      <div class="has-children expanded c-green" id="n12">
                        <div class="node">
                          <span class="eqnum">12</span>
                          <div class="label">Light Limitation</div>
                          <div class="eq">
                            <span class="v">F</span><sub>L</sub>(<span class="v">t</span>) =
                            <span class="frac">
                              <span class="num-f">2.718</span>
                              <span class="den-f">ε(<span class="v">t</span>) <span class="v">H</span></span>
                            </span>
                            <span style="font-size:1.2em;">(</span><span class="v">e</span><sup>−α(<span class="v">t</span>)</sup> − <span class="v">e</span><sup>−<span class="v">I</span><sub>0</sub>(<span class="v">t</span>) / <span class="v">I</span><sub>opt</sub></sup><span style="font-size:1.2em;">)</span>
                            <span class="unit">–</span>
                          </div>
                        </div>
                        <div class="children">
                          <div class="params">
                            <div class="params-title">Parameters of Eq. 12</div>
                            <div class="param-row"><span class="param-sym pt-aux">F<sub>L</sub>(t)</span><span class="param-desc">Light limitation</span><span class="param-unit">–</span></div>
                            <div class="param-row"><span class="param-sym pt-constant">H</span><span class="param-desc">Average depth of the pond</span><span class="param-unit">m</span></div>
                            <div class="param-row"><span class="param-sym pt-aux">ε(t)</span><span class="param-desc">Light extinction</span><span class="param-unit">m⁻¹</span></div>
                            <div class="param-row"><span class="param-sym pt-aux">α(t)</span><span class="param-desc">Specific extinction</span><span class="param-unit">–</span></div>
                            <div class="param-row"><span class="param-sym pt-tab">I₀(t)</span><span class="param-desc">Measured light availability</span><span class="param-unit">μE m⁻² s⁻¹</span></div>
                            <div class="param-row"><span class="param-sym pt-constant">I<sub>opt</sub></span><span class="param-desc">Optimal light availability</span><span class="param-unit">μE m⁻² s⁻¹</span></div>
                          </div>

                          <!-- Children of Eq 12: α (Eq 13) and ε (Eq 14) -->
                          <div class="level">
                            <div class="leaf expanded c-green" id="n13">
                              <div class="node">
                                <span class="eqnum">13</span>
                                <div class="label">Specific Extinction α</div>
                                <div class="eq">
                                  α(<span class="v">t</span>) =
                                  <span class="frac">
                                    <span class="num-f"><span class="v">I</span><sub>0</sub>(<span class="v">t</span>)</span>
                                    <span class="den-f"><span class="v">I</span><sub>opt</sub></span>
                                  </span>
                                  <span class="v">e</span><sup>−ε(<span class="v">t</span>) <span class="v">H</span></sup>
                                  <span class="unit">–</span>
                                </div>
                              </div>
                              <div class="children">
                                <div class="params">
                                  <div class="params-title">Parameters of Eq. 13</div>
                                  <div class="param-row"><span class="param-sym pt-aux">α(t)</span><span class="param-desc">Specific extinction</span><span class="param-unit">–</span></div>
                                  <div class="param-row"><span class="param-sym pt-tab">I₀(t)</span><span class="param-desc">Measured light availability</span><span class="param-unit">μE m⁻² s⁻¹</span></div>
                                  <div class="param-row"><span class="param-sym pt-constant">I<sub>opt</sub></span><span class="param-desc">Optimal light availability</span><span class="param-unit">μE m⁻² s⁻¹</span></div>
                                  <div class="param-row"><span class="param-sym pt-aux">ε(t)</span><span class="param-desc">Light extinction</span><span class="param-unit">m⁻¹</span></div>
                                  <div class="param-row"><span class="param-sym pt-constant">H</span><span class="param-desc">Average depth of the pond</span><span class="param-unit">m</span></div>
                                </div>
                              </div>
                            </div>

                            <div class="leaf expanded c-orange" id="n14">
                              <div class="node">
                                <span class="eqnum">14</span>
                                <div class="label">Light Extinction ε</div>
                                <div class="eq">
                                  ε(<span class="v">t</span>) = ε<sub>0</sub> + λ <span class="v">A</span>(<span class="v">t</span>)
                                  <span class="unit">m⁻¹</span>
                                </div>
                              </div>
                              <div class="children">
                                <div class="params">
                                  <div class="params-title">Parameters of Eq. 14</div>
                                  <div class="param-row"><span class="param-sym pt-aux">ε(t)</span><span class="param-desc">Light extinction</span><span class="param-unit">m⁻¹</span></div>
                                  <div class="param-row"><span class="param-sym pt-constant">ε₀</span><span class="param-desc">Background extinction</span><span class="param-unit">m⁻¹ (= 0.5)</span></div>
                                  <div class="param-row"><span class="param-sym pt-constant">λ</span><span class="param-desc">Algal specific extinction</span><span class="param-unit">m² g⁻¹ (= 1)</span></div>
                                  <div class="param-row"><span class="param-sym pt-state">A(t)</span><span class="param-desc">Algal biomass</span><span class="param-unit">mg L⁻¹</span></div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Macrophytes Experiment - Eq 5 -->
          <div class="has-children expanded c-orange" id="n5">
            <div class="node">
              <span class="eqnum">5</span>
              <div class="label">Macrophytes Experiment</div>
              <div class="eq">
                <span class="v">F</span><sub>mac</sub>(<span class="v">t</span>) =
                <span style="font-size:1.2em;">(</span><span class="v">P</span><sub>m</sub>
                <span class="frac">
                  <span class="num-f"><span class="v">I</span><sub>p</sub>(<span class="v">t</span>)</span>
                  <span class="den-f"><span class="v">I</span><sub>p</sub>(<span class="v">t</span>) + <span class="v">K</span><sub>m</sub></span>
                </span>
                <span class="v">F</span><sub>T2</sub>(<span class="v">t</span>) − <span class="v">R</span><sub>m</sub><span style="font-size:1.2em;">)</span>
                <span class="frac">
                  <span class="num-f"><span class="v">C</span><sub>m</sub> <span class="v">M</span></span>
                  <span class="den-f"><span class="v">H</span></span>
                </span>
                <span class="unit">mg L⁻¹ min⁻¹</span>
              </div>
            </div>
            <div class="children">
              <div class="params">
                <div class="params-title">Parameters of Eq. 5</div>
                <div class="param-row"><span class="param-sym pt-aux">F<sub>mac</sub>(t)</span><span class="param-desc">Macrophytes oxygen production</span><span class="param-unit">mg L⁻¹ min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-constant">P<sub>m</sub></span><span class="param-desc">Maximum oxygen production</span><span class="param-unit">min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-constant">K<sub>m</sub></span><span class="param-desc">Light limitation macrophytes (range 200–500)</span><span class="param-unit">μE m⁻² s⁻¹ (= 250)</span></div>
                <div class="param-row"><span class="param-sym pt-aux">F<sub>T2</sub>(t)</span><span class="param-desc">Temperature correction</span><span class="param-unit">–</span></div>
                <div class="param-row"><span class="param-sym pt-constant">R<sub>m</sub></span><span class="param-desc">Macrophyte oxygen consumption</span><span class="param-unit">min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-constant">C<sub>m</sub></span><span class="param-desc">Fraction area coverage</span><span class="param-unit">–</span></div>
                <div class="param-row"><span class="param-sym pt-constant">M</span><span class="param-desc">Average biomass macrophytes</span><span class="param-unit">g m⁻²</span></div>
                <div class="param-row"><span class="param-sym pt-constant">H</span><span class="param-desc">Average depth of the pond</span><span class="param-unit">m</span></div>
              </div>

              <!-- Children of Eq 5: F_T2 (Eq 9) and I_p (Eq 11) -->
              <div class="level">
                <div class="leaf expanded c-orange" id="n9">
                  <div class="node">
                    <span class="eqnum">9</span>
                    <div class="label">Temperature Correction (macrophytes)</div>
                    <div class="eq">
                      <span class="v">F</span><sub>T2</sub>(<span class="v">t</span>) = Θ<sub>2</sub><sup>(<span class="v">T</span>(<span class="v">t</span>) − 20°C) / γ</sup>
                      <span class="unit">–</span>
                    </div>
                  </div>
                  <div class="children">
                    <div class="params">
                      <div class="params-title">Parameters of Eq. 9</div>
                      <div class="param-row"><span class="param-sym pt-aux">F<sub>T2</sub>(t)</span><span class="param-desc">Temperature limitation</span><span class="param-unit">–</span></div>
                      <div class="param-row"><span class="param-sym pt-constant">Θ₂</span><span class="param-desc">Temperature coefficient macrophytes (1.02–1.06)</span><span class="param-unit">–</span></div>
                      <div class="param-row"><span class="param-sym pt-tab">T(t)</span><span class="param-desc">Temperature (measured)</span><span class="param-unit">°C</span></div>
                      <div class="param-row"><span class="param-sym pt-constant">γ</span><span class="param-desc">Scaling parameter</span><span class="param-unit">°C (= 1)</span></div>
                    </div>
                  </div>
                </div>

                <div class="has-children expanded c-orange" id="n11">
                  <div class="node">
                    <span class="eqnum">11</span>
                    <div class="label">Light at Plant Depth</div>
                    <div class="eq">
                      <span class="v">I</span><sub>p</sub>(<span class="v">t</span>) = <span class="v">I</span><sub>0</sub>(<span class="v">t</span>) <span class="v">e</span><sup>−ε(<span class="v">t</span>) <span class="v">z</span></sup>
                      <span class="unit">μE m⁻² s⁻¹</span>
                    </div>
                  </div>
                  <div class="children">
                    <div class="params">
                      <div class="params-title">Parameters of Eq. 11</div>
                      <div class="param-row"><span class="param-sym pt-aux">I<sub>p</sub>(t)</span><span class="param-desc">Average light availability at plants</span><span class="param-unit">μE m⁻² s⁻¹</span></div>
                      <div class="param-row"><span class="param-sym pt-tab">I₀(t)</span><span class="param-desc">Measured light availability</span><span class="param-unit">μE m⁻² s⁻¹</span></div>
                      <div class="param-row"><span class="param-sym pt-aux">ε(t)</span><span class="param-desc">Light extinction</span><span class="param-unit">m⁻¹</span></div>
                      <div class="param-row"><span class="param-sym pt-constant">z</span><span class="param-desc">Depth to plants</span><span class="param-unit">m</span></div>
                    </div>

                    <!-- Copy of Eq 14 nested here because ε(t) is shared with the macrophytes branch -->
                    <div class="level">
                      <div class="leaf expanded c-orange" id="n14b">
                        <div class="node">
                          <span class="eqnum">14</span>
                          <div class="label">Light Extinction ε</div>
                          <div class="eq">
                            ε(<span class="v">t</span>) = ε<sub>0</sub> + λ <span class="v">A</span>(<span class="v">t</span>)
                            <span class="unit">m⁻¹</span>
                          </div>
                        </div>
                        <div class="children">
                          <div class="params">
                            <div class="params-title">Parameters of Eq. 14</div>
                            <div class="param-row"><span class="param-sym pt-aux">ε(t)</span><span class="param-desc">Light extinction</span><span class="param-unit">m⁻¹</span></div>
                            <div class="param-row"><span class="param-sym pt-constant">ε₀</span><span class="param-desc">Background extinction</span><span class="param-unit">m⁻¹ (= 0.5)</span></div>
                            <div class="param-row"><span class="param-sym pt-constant">λ</span><span class="param-desc">Algal specific extinction</span><span class="param-unit">m² g⁻¹ (= 1)</span></div>
                            <div class="param-row"><span class="param-sym pt-state">A(t)</span><span class="param-desc">Algal biomass</span><span class="param-unit">mg L⁻¹</span></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- SOD Experiment - Eq 6 -->
          <div class="leaf expanded c-grey" id="n6">
            <div class="node">
              <span class="eqnum">6</span>
              <div class="label">Sediment Oxygen Demand Experiment</div>
              <div class="eq">
                <span class="v">F</span><sub>SOD</sub> =
                <span class="frac">
                  <span class="num-f"><span class="v">SOD</span></span>
                  <span class="den-f"><span class="v">H</span></span>
                </span>
                <span class="unit">mg L⁻¹ min⁻¹</span>
              </div>
            </div>
            <div class="children">
              <div class="params">
                <div class="params-title">Parameters of Eq. 6</div>
                <div class="param-row"><span class="param-sym pt-aux">F<sub>SOD</sub></span><span class="param-desc">Sediment oxygen demand (per volume)</span><span class="param-unit">mg L⁻¹ min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-constant">SOD</span><span class="param-desc">Sediment oxygen demand (per area)</span><span class="param-unit">mg m⁻² min⁻¹</span></div>
                <div class="param-row"><span class="param-sym pt-constant">H</span><span class="param-desc">Average depth of the pond</span><span class="param-unit">m</span></div>
              </div>
            </div>
          </div>
        </div>

        <!-- Measurement leaves (Tabulars) -->
        <div class="level">
          <div class="leaf expanded c-pink" id="n15">
            <div class="node">
              <span class="eqnum">15</span>
              <div class="label">Continuous oxygen monitoring</div>
              <div class="eq">
                <span class="v">O</span><sub>2</sub>(<span class="v">t</span>)<sub>obs</sub>
                <span class="unit">mg L⁻¹</span>
              </div>
            </div>
            <div class="children">
              <div class="params">
                <div class="params-title">Measurement</div>
                <div class="param-row"><span class="param-sym pt-tab">O₂(t)_obs</span><span class="param-desc">Observed (measured) oxygen concentration</span><span class="param-unit">mg L⁻¹</span></div>
              </div>
            </div>
          </div>

          <div class="leaf expanded c-pink" id="nI0">
            <div class="node">
              <span class="eqnum">I</span>
              <div class="label">Continuous light availability monitoring</div>
              <div class="eq">
                <span class="v">I</span><sub>0</sub>(<span class="v">t</span>)
                <span class="unit">μE m⁻² s⁻¹</span>
              </div>
            </div>
            <div class="children">
              <div class="params">
                <div class="params-title">Measurement</div>
                <div class="param-row"><span class="param-sym pt-tab">I₀(t)</span><span class="param-desc">Measured incoming light availability</span><span class="param-unit">μE m⁻² s⁻¹</span></div>
              </div>
            </div>
          </div>

          <div class="leaf expanded c-pink" id="nT">
            <div class="node">
              <span class="eqnum">T</span>
              <div class="label">Continuous temperature monitoring</div>
              <div class="eq">
                <span class="v">T</span>(<span class="v">t</span>)
                <span class="unit">°C</span>
              </div>
            </div>
            <div class="children">
              <div class="params">
                <div class="params-title">Measurement</div>
                <div class="param-row"><span class="param-sym pt-tab">T(t)</span><span class="param-desc">Measured water temperature</span><span class="param-unit">°C</span></div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</div>

</body>
</html>
c.html…]()



