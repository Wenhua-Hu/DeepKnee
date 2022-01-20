$(function () {

    let metrics_data = {
        'resnet18': {
            '0': [
                {product: 'KL 0', 'Accurancy': 94.4, 'Confidence': 60.3},
                {product: 'KL 1', 'Accurancy': 7.1, 'Confidence': 27.5},
                {product: 'KL 2', 'Accurancy': 60.9, 'Confidence': 42.9},
                {product: 'KL 3', 'Accurancy': 68.7, 'Confidence': 57.2},
                {product: 'KL 4', 'Accurancy': 72.6, 'Confidence': 56.3}
            ],
            '1': [
                {product: 'KL 0', '0': 60.29, '1': 24.57, '2': 13.45, '3': 1.57, '4': 0.12},
                {product: 'KL 1', '0': 44.14, '1': 27.5, '2': 24.76, '3': 3.37, '4': 0.22},
                {product: 'KL 2', '0': 24.73, '1': 23.34, '2': 42.83, '3': 8.5, '4': 0.6},
                {product: 'KL 3', '0': 5.24, '1': 8.83, '2': 24.97, '3': 57.13, '4': 3.84},
                {product: 'KL 4', '0': 2.21, '1': 1.84, '2': 8.44, '3': 31.25, '4': 56.25}
            ]
        },
        'vgg16': {
            '0': [
                {product: 'KL 0', 'Accurancy': 64.48, 'Confidence': 39.49},
                {product: 'KL 1', 'Accurancy': 29.06, 'Confidence': 26.55},
                {product: 'KL 2', 'Accurancy': 77.86, 'Confidence': 37.24},
                {product: 'KL 3', 'Accurancy': 60.54, 'Confidence': 45.32},
                {product: 'KL 4', 'Accurancy': 60.79, 'Confidence': 42.66},

            ],
            '1': [
                {product: 'KL 0', '0': 39.49, '1': 26.58, '2': 17.81, '3': 10.03, '4': 6.09},
                {product: 'KL 1', '0': 23.17, '1': 26.55, '2': 28.28, '3': 14.01, '4': 7.99},
                {product: 'KL 2', '0': 12.72, '1': 20.67, '2': 37.24, '3': 18.8, '4': 10.56},
                {product: 'KL 3', '0': 7.25, '1': 11.33, '2': 23.25, '3': 45.32, '4': 12.84},
                {product: 'KL 4', '0': 5.49, '1': 7.28, '2': 9.55, '3': 35.02, '4': 42.66},

            ]
        },
        'vgg19': {
            '0': [
                {product: 'KL 0', 'Accurancy': 69.18, 'Confidence': 51.33},
                {product: 'KL 1', 'Accurancy': 20.28, 'Confidence': 22.99},
                {product: 'KL 2', 'Accurancy': 84.79, 'Confidence': 52.68},
                {product: 'KL 3', 'Accurancy': 55.16, 'Confidence': 42.68},
                {product: 'KL 4', 'Accurancy': 86.28, 'Confidence': 74.42},

            ],
            '1': [
                {product: 'KL 0', '0': 51.33, '1': 20.29, '2': 16.84, '3': 7.17, '4': 4.38},
                {product: 'KL 1', '0': 22.75, '1': 22.99, '2': 37.19, '3': 10.71, '4': 6.36},
                {product: 'KL 2', '0': 10.93, '1': 15.31, '2': 52.68, '3': 13.12, '4': 7.96},
                {product: 'KL 3', '0': 6.66, '1': 9.44, '2': 26.71, '3': 42.68, '4': 14.51},
                {product: 'KL 4', '0': 1.78, '1': 2.61, '2': 5.18, '3': 16, '4': 74.42},

            ]
        },
        'resnet34': {
            '0': [
                {product: 'KL 0', 'Accurancy': 69.96, 'Confidence': 38.89},
                {product: 'KL 1', 'Accurancy': 0.68, 'Confidence': 21.41},
                {product: 'KL 2', 'Accurancy': 90.83, 'Confidence': 40.7},
                {product: 'KL 3', 'Accurancy': 54.71, 'Confidence': 46.24},
                {product: 'KL 4', 'Accurancy': 60.79, 'Confidence': 42.86},

            ],
            '1': [
                {product: 'KL 0', '0': 38.89, '1': 19.05, '2': 23.59, '3': 12.09, '4': 6.39},
                {product: 'KL 1', '0': 24.21, '1': 21.41, '2': 31.94, '3': 14.94, '4': 7.49},
                {product: 'KL 2', '0': 14.28, '1': 19.13, '2': 40.7, '3': 17.69, '4': 8.2},
                {product: 'KL 3', '0': 6.01, '1': 11.11, '2': 29.1, '3': 46.24, '4': 7.54},
                {product: 'KL 4', '0': 4.27, '1': 5.76, '2': 11.02, '3': 36.09, '4': 42.86},

            ]
        },
        'resnet50': {
            '0': [
                {product: 'KL 0', 'Accurancy': 67.14, 'Confidence': 46.75},
                {product: 'KL 1', 'Accurancy': 25.34, 'Confidence': 22.13},
                {product: 'KL 2', 'Accurancy': 75.84, 'Confidence': 47.27},
                {product: 'KL 3', 'Accurancy': 68.61, 'Confidence': 59.3},
                {product: 'KL 4', 'Accurancy': 39.22, 'Confidence': 40.07},

            ],
            '1': [
                {product: 'KL 0', '0': 46.75, '1': 21.1, '2': 14.16, '3': 12.03, '4': 5.96},
                {product: 'KL 1', '0': 23.17, '1': 22.13, '2': 29.46, '3': 17.48, '4': 7.75},
                {product: 'KL 2', '0': 8.76, '1': 14.23, '2': 47.27, '3': 21.2, '4': 8.55},
                {product: 'KL 3', '0': 2.13, '1': 6.13, '2': 24.56, '3': 59.3, '4': 7.88},
                {product: 'KL 4', '0': 2.3, '1': 2.93, '2': 4.15, '3': 50.55, '4': 40.07},

            ]
        },
        'resnet101': {
            '0': [
                {product: 'KL 0', 'Accurancy': 73.09, 'Confidence': 45.21},
                {product: 'KL 1', 'Accurancy': 14.53, 'Confidence': 22.76},
                {product: 'KL 2', 'Accurancy': 81.21, 'Confidence': 39.9},
                {product: 'KL 3', 'Accurancy': 47.99, 'Confidence': 38.5},
                {product: 'KL 4', 'Accurancy': 82.36, 'Confidence': 69.81},

            ],
            '1': [
                {product: 'KL 0', '0': 45.21, '1': 20.7, '2': 16.39, '3': 10.7, '4': 7},
                {product: 'KL 1', '0': 27.4, '1': 22.76, '2': 27.14, '3': 13.93, '4': 8.77},
                {product: 'KL 2', '0': 13.92, '1': 17.97, '2': 39.9, '3': 17.49, '4': 10.72},
                {product: 'KL 3', '0': 6.7, '1': 11.08, '2': 29.43, '3': 38.5, '4': 14.29},
                {product: 'KL 4', '0': 2.65, '1': 2.96, '2': 6.03, '3': 18.56, '4': 69.81},

            ]
        },
    }


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
            source: metrics_data['resnet18']['0']
        },
        xAxis: {type: 'category'},
        yAxis: {

            type: 'value',
            min: 0,
            max: 100,
            position: 'left',
            axisLabel: {
                formatter: '{value} %'
            }

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
            source: metrics_data['resnet18']['1']
        },
        xAxis: {type: 'category'},
        yAxis: {
            type: 'value',
            min: 0,
            max: 100,
            position: 'left',
            axisLabel: {
                formatter: '{value} %'
            }
        },
        // Declare several bar series, each will be mapped
        // to a column of dataset.source by default.
        series: [{type: 'bar'}, {type: 'bar'}, {type: 'bar'}, {type: 'bar'}, {type: 'bar'}]
    };

    Chart2.setOption(option2);


});





