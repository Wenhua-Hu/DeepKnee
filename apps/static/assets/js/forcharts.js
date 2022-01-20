$(function () {
    let data_resnet18_0 = [
        {product: 'KL 0', 'Accurancy': 94.4, 'Confidence': 60.3},
        {product: 'KL 1', 'Accurancy': 7.1, 'Confidence': 27.5},
        {product: 'KL 2', 'Accurancy': 60.9, 'Confidence': 42.9},
        {product: 'KL 3', 'Accurancy': 68.7, 'Confidence': 57.2},
        {product: 'KL 4', 'Accurancy': 72.6, 'Confidence': 56.3}
    ]


    let data_resnet18_1 = [
        {product: 'KL 0', '0': 60.29, '1': 24.57,'2': 13.45, '3':1.57,'4': 0.12},
        {product: 'KL 1', '0': 44.14, '1':27.5,'2': 24.76, '3': 3.37,'4': 0.22},
        {product: 'KL 2', '0': 24.73, '1': 23.34,'2': 42.83, '3': 8.5,'4': 0.6},
        {product: 'KL 3', '0': 5.24, '1': 8.83,'2': 24.97, '3': 57.13,'4': 3.84},
        {product: 'KL 4', '0': 2.21, '1': 1.84,'2': 8.44, '3': 31.25,'4': 56.25}
    ]




    var Chart1 = echarts.init(document.getElementById('chart_1'));
    var option1;

    option1 = {
        legend: {
            orient: 'horizontal',
            // right: 20,
            // top: 'top'
        },
        tooltip: {},
        dataset: {
            dimensions: ['product', 'Accurancy', 'Confidence'],
            source: data_resnet18_0
        },
        xAxis: {type: 'category'},
        yAxis: {
            type: 'value',
            name: 'Percentage'
            // ...
        },
        // Declare several bar series, each will be mapped
        // to a column of dataset.source by default.
        series: [{type: 'bar'}, {type: 'bar'}]
    };

    Chart1.setOption(option1);


    var Chart2 = echarts.init(document.getElementById('chart_2'));
    var option2;


    option2 = {
        legend: {
            orient: 'horizontal',
            // right: 0,
            // top: 'center'
        },
        tooltip: {},
        dataset: {
            dimensions: ['product', '0', '1', '2', '3', '4'],
            source: data_resnet18_1
        },
        xAxis: {type: 'category'},
        yAxis: {
            type: 'value',
            name: 'Confidence %'
            // ...
        },
        // Declare several bar series, each will be mapped
        // to a column of dataset.source by default.
        series: [{type: 'bar'}, {type: 'bar'},{type: 'bar'}, {type: 'bar'},{type: 'bar'}]
    };

    Chart2.setOption(option2);


});



