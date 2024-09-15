$(document).ready(function() {
    $('form').submit(function(e) {
        e.preventDefault(); // Prevent form submission
        var inputText = $('textarea[name="input_text"]').val();
        $.post('/', { input_text: inputText }, function(data) {
            $('#output_text').val(data.processed_text);
        });
    });
});