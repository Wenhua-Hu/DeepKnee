
var chartDom = document.getElementById('chart_1');
var Chart1 = echarts.init(chartDom);
var option1;

option1 = {
  legend: {},
  tooltip: {},
  dataset: {
    dimensions: ['product', 'Prediction', 'Confidence'],
    source: [
      { product: 'KL 0', Prediction: 0, Confidence: 85.8},
      { product: 'KL 1', Prediction: 0, Confidence: 73.4},
      { product: 'KL 2', Prediction: 0, Confidence: 65.2},
      { product: 'KL 3', Prediction: 0, Confidence: 53.9},
      { product: 'KL 4', Prediction: 0, Confidence: 53.9},
    ]
  },
  xAxis: { type: 'category' },
  yAxis: {},
  // Declare several bar series, each will be mapped
  // to a column of dataset.source by default.
  series: [ { type: 'bar' }, { type: 'bar' }]
};

Chart1.setOption(option1);





var chartDom = document.getElementById('chart_2');
var Chart2 = echarts.init(chartDom);
var option2;



option2 = {
  legend: {},
  tooltip: {},
  dataset: {
    dimensions: ['product', 'Accurancy'],
    source: [
      { product: 'V-16', Accurancy: 43},
      { product: 'V-19', Accurancy: 83},
      { product: 'R-18', Accurancy: 86},
      { product: 'R-34', Accurancy: 72},
      { product: 'R-50', Accurancy: 72},
      { product: 'R-101', Accurancy: 72},
    ]
  },
  xAxis: { type: 'category' },
  yAxis: {},
  // Declare several bar series, each will be mapped
  // to a column of dataset.source by default.
  series: [ { type: 'bar' },]
};

Chart2.setOption(option2);