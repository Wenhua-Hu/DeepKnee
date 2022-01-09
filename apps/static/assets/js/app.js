$(function () {
    $(".card.card-image-1").click(function () {
            url = $(this).attr('pic_url');
            xray = $(this).attr('xray');

            if (xray == 'L') {
                $('#ap-1').attr('src', url);
                $('#ap-pop-1').attr('src', url);
            } else if (xray == 'R') {
                $('#pa-1').attr('src', url);
                $('#pa-pop-1').attr('src', url);
            }
        }
    );

    $(".card.card-image-1").click(function () {
        image_name = $(this).attr('filename');
        console.log(image_name)
        $.post("/predict", {filename: image_name}, function (data) {
            predictions = data['prediction'];
            predicted_label = data['predicted_label'];
            heatmap_1 = data['heatmap_1']

            pred_0 = predictions[0];
            pred_1 = predictions[1];
            pred_2 = predictions[2];
            pred_3 = predictions[3];
            pred_4 = predictions[4];
            $("#r0").text((pred_0* 100).toFixed(2));
            $("#r1").text((pred_1* 100).toFixed(2));
            $("#r2").text((pred_2* 100).toFixed(2));
            $("#r3").text((pred_3 * 100).toFixed(2));
            $("#r4").text((pred_4 * 100).toFixed(2));
            $("#predicted_label").text(predicted_label);
            $('#ht_1').attr('src', './static/assets/images/gradcam/'+heatmap_1);
            $('#ht1_1').attr('src', './static/assets/images/gradcam/'+heatmap_1);

        }, "json");
    });


    //
    //  $('.card.card-image-1').bind('click', function() {
    //   $.getJSON('/predict', {
    //
    //     filename: $(this).attr('filename')
    //   }, function(data) {
    //     predictions = data['prediction'];
    //     pred_0 = predictions[0];
    //     pred_1 = predictions[1];
    //     pred_2 = predictions[2];
    //     pred_3 = predictions[3];
    //     pred_4 = predictions[4];
    //      // $('#tr_'1 + ' td:nth-child(5)').text('Renewed');
    //     $("#r0").text(pred_0*100);
    //     $("#r1").text(pred_1*100);
    //     $("#r2").text(pred_2*100);
    //     $("#r3").text(pred_3*100);
    //     $("#r4").text(pred_4*100);
    //
    //   });
    //   return false;
    // });

    // $("#search_button").click(function() {
    //     var search_word = $("#search_box").val();
    //     console.log(search_word)
    //     var dataString = 'search_word='+ search_word;
    //     console.log(dataString)
    //     if(search_word==''){
    //     }else{
    //       $.ajax({
    //         type: "POST",
    //         url: "/searchdata",
    //         data: dataString,
    //         cache: false,
    //         success: function(data){
    //             $("input:text").val(data["patient"]['name']);
    //             $('#patient_id').text(data["patient"]['id']);
    //             $('#patient_name').text(data["patient"]['name']);
    //             $('#patient_id').text(data["patient"]['birthdate']);
    //             image_url = "./static/assets/images/knee/"
    //             images = data['images']
    //
    //
    //         }
    //       });
    //     }
    //   return false;
    // });
});
