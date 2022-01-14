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
        xray = $(this).attr('xray');

        $.post("/predict", {filename: image_name}, function (data) {
            probas = data['prediction'];
            stage = data['predicted_label'];
            heatmap_urls = data['heatmaps_url']

            $('table.prediction-proba td span').each(function (k, element) {
                $(element).text((probas[k] * 100).toFixed(2))
            });

            $(".prediction-stage a:nth-child(2)").text(stage);

            $('.card-thumbnail img').each(function (k, element) {
                if (heatmap_urls[k] !== undefined) {
                    $(element).attr('src', './static/assets/images/gradcam/' + heatmap_urls[k]);
                }
            });

            if (xray == 'L') {
                $('.card-image-l a.card-image-hm').each(function (k, element) {
                    if (heatmap_urls[k] !== undefined) {
                        $(element).find('img').attr('src', './static/assets/images/gradcam/' + heatmap_urls[k]);
                    }
                });

                $('.card-image-l div.card-image-hm-pop').each(function (k, element) {
                    if (heatmap_urls[k] !== undefined) {
                        $(element).find('img').attr('src', './static/assets/images/gradcam/' + heatmap_urls[k]);
                    }
                });

            } else if (xray == 'R') {
                $('.card-image-r a.card-image-hm').each(function (k, element) {
                    if (heatmap_urls[k] !== undefined) {
                        $(element).find('img').attr('src', './static/assets/images/gradcam/' + heatmap_urls[k]);
                    }
                });

                $('.card-image-r div.card-image-hm-pop').each(function (k, element) {
                    if (heatmap_urls[k] !== undefined) {
                        $(element).find('img').attr('src', './static/assets/images/gradcam/' + heatmap_urls[k]);
                    }
                });

            }

        }, "json");
    });



});
