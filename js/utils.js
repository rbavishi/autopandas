function warn(message) {
    $('#alerts').append(
        '<div class="alert alert-dismissible alert-danger autopandas-warning">' +
        '<button type="button" class="close" data-dismiss="alert">' +
        '&times;</button>' + message + '</div>');

    setTimeout(function() {
        $('.autopandas-warning').remove();
    }, 5000);
}

function custom_waiting_logo(msg) {
    return $('<p><span class=\"fas fa-spin fa-spinner\"></span> ' + msg + '</p>');
}