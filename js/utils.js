function warn(message) {
    let element_volatile = $('<div class="alert alert-dismissible alert-danger autopandas-warning">' +
        '<button type="button" class="close" data-dismiss="alert">' +
        '&times;</button>' + message + '</div>');

    let element = $('<div class="alert alert-dismissible alert-danger autopandas-warning-persistent">' +
        '<button type="button" class="close" data-dismiss="alert">' +
        '&times;</button>' + message + '</div>');

    $('#alerts').append(element_volatile);

    setTimeout(function() {
        $('.autopandas-warning').remove();
    }, 5000);

    $('#autopandas-log').prepend(element);
}

function success_alert(message) {
    let element_volatile = $('<div class="alert alert-dismissible alert-success autopandas-warning">' +
        '<button type="button" class="close" data-dismiss="alert">' +
        '&times;</button>' + message + '</div>');

    let element = $('<div class="alert alert-dismissible alert-success autopandas-warning-persistent">' +
        '<button type="button" class="close" data-dismiss="alert">' +
        '&times;</button>' + message + '</div>');

    $('#alerts').append(element_volatile);
    setTimeout(function() {
        $('.autopandas-warning').remove();
    }, 5000);

    $('#autopandas-log').prepend(element);
}

function custom_waiting_logo(msg) {
    return $('<p><span class=\"fas fa-spin fa-spinner\"></span> ' + msg + '</p>');
}