// Tiny factory that wires `react-plotly.js` to the lightweight dist-min bundle.
// Using this keeps our production bundle around 3-4 MB smaller than the full plotly.js.
import createPlotlyComponent from 'react-plotly.js/factory';
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore - plotly.js-dist-min has no type defs
import Plotly from 'plotly.js-dist-min';

const Plot = createPlotlyComponent(Plotly);

export default Plot;
