$(function () {
    $(".card.card-image-1").click(function () {
        url = $(this).attr('pic_url');
        LR = $(this).attr('xray');
        src= './static/assets/images/default.png'

        $('.card-thumbnail img').attr('src', src);

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

        const loadText = document.querySelector(".loading-text");

        let load = 0;
        let int = setInterval(blurring,n_samples*3);

        function blurring(){
          load++
          if(load>98){
            clearInterval(int);
          }
          loadText.innerText = `${load}%`
          // loadText.style.opacity = scale(load, 0, 100, 1, 0)
        }
        const scale = (num, in_min, in_max, out_min, out_max) => {
          return ((num - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min
        }

        $.post("/predict_lime", {filename: image_name, modelname: model_name, nsamples: n_samples}, function(data) {
            lime_path = data['output_lime']
            $('.card-thumbnail img:eq(2)').attr('src', './static/assets/images/knee_lime/' + lime_path);
            // const loadText = document.querySelector(".loading-text");
            // loadText.style.opacity = scale(100, 0, 100, 1, 0)
        }, "json");
    });


    $(".card-thumbnail img").click(function () {

            LR = $("#infer-thumbnail").attr('xray');
            console.log(LR);
            src = $(this).attr('src');
            console.log(src);
            index_ = $(this).index(".card-thumbnail img");
            console.log(index_)

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

