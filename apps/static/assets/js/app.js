$(function () {
    $(".card.card-image-1").click(function () {

            url = $(this).attr('pic_url');
            xray = $(this).attr('xray');

            if (xray == 'L') {
                $('.card-image-l a.card-image-org img').attr('src', url);
                $('.card-image-l div.card-image-org-pop img').attr('src', url);
            } else if (xray == 'R') {
                $('.card-image-r a.card-image-org img').attr('src', url);
                $('.card-image-r div.card-image-org-pop img').attr('src', url);
            }
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
            heapmap_path = data['output']
            $('.card-thumbnail img:first').attr('src',  './static/assets/images/knee_gradcam/' + heapmap_path);

            if (LR == 'L') {
                $('.card-image-l a.card-image-hm:first').find('img').attr('src', './static/assets/images/knee_gradcam/' + heapmap_path);
                $('.card-image-l div.card-image-hm-pop:first').find('img').attr('src', './static/assets/images/knee_gradcam/' + heapmap_path);

            } else if (LR == 'R') {
                $('.card-image-r a.card-image-hm:first').find('img').attr('src', './static/assets/images/knee_gradcam/' + heapmap_path);
                $('.card-image-r div.card-image-hm-pop:first').find('img').attr('src', './static/assets/images/knee_gradcam/' + heapmap_path);

            }

        }, "json");
    });


    $(".card.card-image-1").click(function () {

        image_name = $(this).attr('filename');
        model_name = $('input[name=models]:checked').val();
        LR = $(this).attr('xray');

        $.post("/predict_lime", {filename: image_name, modelname: model_name}, function (data) {
            lime_path = data['output_lime']
            $('.card-thumbnail img:last').attr('src',  './static/assets/images/knee_lime/' + lime_path);

            if (LR == 'L') {
                $('.card-image-l a.card-image-hm:last').find('img').attr('src', './static/assets/images/knee_lime/' + lime_path);
                $('.card-image-l div.card-image-hm-pop:last').find('img').attr('src', './static/assets/images/knee_lime/' + lime_path);

            } else if (LR == 'R') {
                $('.card-image-r a.card-image-hm:last').find('img').attr('src', './static/assets/images/knee_lime/' + lime_path);
                $('.card-image-r div.card-image-hm-pop:last').find('img').attr('src', './static/assets/images/knee_lime/' + lime_path);

            }

        }, "json");
    });





});
