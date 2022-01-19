$(function () {
    $(".card.card-image-1").click(function () {
            url = $(this).attr('pic_url');
            LR = $(this).attr('xray');
            src = './static/assets/images/default.png';


            $('.card-thumbnail img').attr('src', src);
            $("div#feedback .heatmap").text("HEATMAP 1");

            if (LR == 'L') {
                $('.card-image-l a.card-image-hm').each(function (k, element) {
                    $(element).find('img').attr('src', src);
                });

                $('.card-image-l div.card-image-hm-pop').each(function (k, element) {
                    $(element).find('img').attr('src', src);
                });

                $('.card-image-l a.card-image-org img').attr('src', url);
                $('.card-image-l div.card-image-org-pop img').attr('src', url);
            } else if (LR == 'R') {
                $('.card-image-r a.card-image-hm').each(function (k, element) {
                    $(element).find('img').attr('src', src);
                });
                $('.card-image-r div.card-image-hm-pop').each(function (k, element) {
                    $(element).find('img').attr('src', src);
                });

                $('.card-image-r a.card-image-org img').attr('src', url);
                $('.card-image-r div.card-image-org-pop img').attr('src', url);
            }

            $('#infer-thumbnail').attr('xray', LR);
            $('#image-explainer').attr('xray', LR);


        }
    );


    $(".card.card-image-1").click(function () {

        image_name = $(this).attr('filename');
        model_name = $('input[name=models]:checked').val();

        $.post("/predict_score", {filename: image_name, modelname: model_name}, function (data) {
            probas = data['prediction'];
            stage = data['predicted_label'];

            $('table.prediction-proba td span').each(function (k, element) {
                $(element).text((probas[k] * 100).toFixed(2))
            });

            $(".prediction-stage a:nth-child(2)").text(stage);


            var myChart = echarts.init(document.getElementById('chart_1'));
            option1 = {
                  legend: {},
                  tooltip: {},
                  dataset: {
                    dimensions: ['product', 'Prediction', 'Confidence'],
                    source: [
                      { product: 'KL 0', Prediction: parseInt(data['prediction'][0]*100).toFixed(2), Confidence: 85.8},
                      { product: 'KL 1', Prediction: parseInt(data['prediction'][1]*100).toFixed(2), Confidence: 73.4},
                      { product: 'KL 2', Prediction: parseInt(data['prediction'][2]*100).toFixed(2), Confidence: 65.2},
                      { product: 'KL 3', Prediction: parseInt(data['prediction'][3]*100).toFixed(2), Confidence: 53.9},
                      { product: 'KL 4', Prediction: parseInt(data['prediction'][4]*100).toFixed(2), Confidence: 53.9},
                    ]
                  },
                  xAxis: { type: 'category' },
                  yAxis: {},
                  // Declare several bar series, each will be mapped
                  // to a column of dataset.source by default.
                  series: [ { type: 'bar' }, { type: 'bar' }]
                };
            myChart.setOption(option1);


        }, "json");
    });


    $(".card.card-image-1").click(function () {

        image_name = $(this).attr('filename');
        model_name = $('input[name=models]:checked').val();
        LR = $(this).attr('xray');

        $.post("/predict_gradcam", {filename: image_name, modelname: model_name}, function (data) {
            heapmap_path = data['output_gradcam']
            boundingbox_path = data['output_bbox']

            $('.card-thumbnail img:eq(0)').attr('src', './static/assets/images/knee_gradcam/' + heapmap_path);
            $('.card-thumbnail img:eq(1)').attr('src', './static/assets/images/knee_boundingbox/' + boundingbox_path);


            if (LR == 'L') {
                $('.card-image-l a.card-image-hm:eq(0)').find('img').attr('src', './static/assets/images/knee_gradcam/' + heapmap_path);
                $('.card-image-l div.card-image-hm-pop:eq(0)').find('img').attr('src', './static/assets/images/knee_gradcam/' + heapmap_path);

                $('.card-image-l a.card-image-hm:eq(1)').find('img').attr('src', './static/assets/images/knee_boundingbox/' + boundingbox_path);
                $('.card-image-l div.card-image-hm-pop:eq(1)').find('img').attr('src', './static/assets/images/knee_boundingbox/' + boundingbox_path);

            } else if (LR == 'R') {
                $('.card-image-r a.card-image-hm:eq(0)').find('img').attr('src', './static/assets/images/knee_gradcam/' + heapmap_path);
                $('.card-image-r div.card-image-hm-pop:eq(0)').find('img').attr('src', './static/assets/images/knee_gradcam/' + heapmap_path);

                $('.card-image-r a.card-image-hm:eq(1)').find('img').attr('src', './static/assets/images/knee_boundingbox/' + boundingbox_path);
                $('.card-image-r div.card-image-hm-pop:eq(1)').find('img').attr('src', './static/assets/images/knee_boundingbox/' + boundingbox_path);

            }

        }, "json");
    });


    $(".card.card-image-1").click(function () {

        image_name = $(this).attr('filename');
        model_name = $('input[name=models]:checked').val();


        LR = $(this).attr('xray');

        let n_samples = 100;
        let load = 0;

        if (model_name == 'resnet18') {
            times = n_samples * 1.2
        } else if (model_name == 'resnet34') {
            times = n_samples * 2
        } else if (model_name == 'resnet50') {
            times = n_samples * 3
        } else if (model_name == 'resnet101') {
            times = n_samples * 4
        } else if (model_name == 'resnet152') {
            times = n_samples * 6
        } else if (model_name == 'vgg16') {
            console.log(model_name)
            times = n_samples * 14
        } else if (model_name == 'vgg19') {
            times = n_samples * 21
        }

        const loadText = document.querySelector(".loading-text");
        const scale = (num, in_min, in_max, out_min, out_max) => {
            return ((num - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min
        }

        loadText.innerText = `0%`
        loadText.style.opacity = scale(0, 0, 100, 1, 0)

        let int = setInterval(blurring, times);
        console.log(int);

        function blurring() {
            load++
            if (load > 99) {
                clearInterval(int);
            }
            loadText.innerText = `${load}%`
            loadText.style.opacity = scale(load, 0, 100, 1, 0)
        }


        $.post("/predict_lime", {filename: image_name, modelname: model_name, nsamples: n_samples}, function (data) {
            lime_path = data['output_lime']
            $('.card-thumbnail img:eq(2)').attr('src', './static/assets/images/knee_lime/' + lime_path);
            const loadText = document.querySelector(".loading-text");
            loadText.style.opacity = scale(100, 0, 100, 1, 0);
            load = 99

        }, "json");
    });


    $("div.card-thumbnail div.card").click(function () {

            LR = $("#infer-thumbnail").attr('xray');

            src = $(this).children("img").attr('src');

            index_ = $(this).index("div.card-thumbnail div.card");
            num = parseInt(index_) + 1;
            $("div#feedback .heatmap").text("HEATMAP " + num);

            if (LR == 'L') {
                if (index_ >= 2) {
                    $('.card-image-l a.card-image-hm:last').find('img').attr('src', src);
                    $('.card-image-l div.card-image-hm-pop:last').find('img').attr('src', src);
                } else {
                    $('.card-image-l a.card-image-hm:eq(' + parseInt(index_) + ')').find('img').attr('src', src);
                    $('.card-image-l div.card-image-hm-pop:eq(' + parseInt(index_) + ')').find('img').attr('src', src);
                }
            } else if (LR == 'R') {
                if (index_ >= 2) {
                    $('.card-image-r a.card-image-hm:last').find('img').attr('src', src);
                    $('.card-image-r div.card-image-hm-pop:last').find('img').attr('src', src);
                } else {
                    $('.card-image-r a.card-image-hm:eq(' + parseInt(index_) + ')').find('img').attr('src', src);
                    $('.card-image-r div.card-image-hm-pop:eq(' + parseInt(index_) + ')').find('img').attr('src', src);
                }
            }
        }
    );


});

