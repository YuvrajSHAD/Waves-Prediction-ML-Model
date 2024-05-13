$(document).ready(function() {
    $('#loading-overlay').hide();
});
document.addEventListener('DOMContentLoaded', function() {
    var video = document.getElementById('video-background');
    setTimeout(function() {
        fadeOut(video);
    }, 3000); // Call fadeOut function after 4 seconds
});

function fadeOut(element) {
    element.style.transition = "opacity 2s"; // Adjust the duration as needed
    element.style.opacity = 0;
    setTimeout(function() {
        element.pause(); // Pause the video
        element.style.display = 'none'; // Hide the video
    }, 1000); // Adjust the delay to match the transition duration
}
document.getElementById('blur-btn').addEventListener('click', function () {
        // Send AJAX request to Flask server
        $('#loading-overlay').show();
        $('body').addClass('blur');
        $.ajax({
            url: "/triggerml",
            type: "POST",
            success: function(response) {
                // Handle successful response
                console.log(response);
                
                $('#loading-overlay').hide();
                $('body').removeClass('blur');
                // Display train and test errors on the HTML page
                document.getElementById('train-rmse-hs').innerText =  response.train_errors[0];
                document.getElementById('train-rmse-hmax').innerText = response.train_errors[1];
                document.getElementById('train-rmse-tz').innerText =  response.train_errors[2];
                document.getElementById('train-rmse-tp').innerText =  response.train_errors[3];
                document.getElementById('train-rmse-dir').innerText =  response.train_errors[4];
                document.getElementById('train-rmse-sst').innerText =  response.train_errors[5];

                document.getElementById('test-rmse-hs').innerText =  response.test_errors[0];
                document.getElementById('test-rmse-hmax').innerText = response.test_errors[1];
                document.getElementById('test-rmse-tz').innerText =  response.test_errors[2];
                document.getElementById('test-rmse-tp').innerText =  response.test_errors[3];
                document.getElementById('test-rmse-dir').innerText =  response.test_errors[4];
                document.getElementById('test-rmse-sst').innerText =  response.test_errors[5];
                document.body.style.overflow = 'auto';

                alert("ML code triggered successfully!");
            },
            error: function(xhr, status, error) {
                // Handle error response
                console.error(xhr.responseText);
                alert("Error: " + xhr.responseText);
                $('#loading-overlay').hide();
                $('body').removeClass('blur');
            }
        });
    });
