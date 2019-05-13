function makeEditor(pycode) {
    let editor = ace.edit(pycode, {
        theme: "ace/theme/crimson_editor",
        mode: "ace/mode/python",
        fontSize: "1em",
        maxLines: Infinity,
        wrap: true
    });

    $.data(pycode, "editor", editor);
}

function makeUserInputEditor(pycode, preview_link) {
    makeEditor(pycode);
    let editor = $.data(pycode, "editor");
    $.data(preview_link, "prevCodeValue", editor.getValue());
    $(preview_link).click(() => {
        let code = editor.getValue();
        if (code === $.data(preview_link, "prevCodeValue")) return;
        $.data(preview_link, "prevCodeValue", code);

        //  Adding a loading sign for clarity
        let parentId = $(pycode).parent().parent().attr('id');
        let previewElem = $('#iopreview' + parentId.substr(-1));
        previewElem.empty();
        previewElem.append(custom_waiting_logo('Loading'));

        let eval_query = ajax_lambda({task: 'evalcode', input: code});
        eval_query.done(function (msg) {
            makePreview(previewElem[0], msg.dtype, msg.data);
            let previewElemList = $('#a-iopreview' + parentId.substr(-1));
            previewElemList.tab('show');

        }).fail(function (error) {
            let log = "";
            if (error.responseJSON) {
                log = "Encountered Error : " + error.responseJSON['log'];
            } else {
                log = "Something went wrong. Check your code";
            }

            warn(log);
            previewElem.empty();
            previewElem.append($('<div class="alert alert-dismissible alert-danger">' +
                '<p>' + log + '</p>' +
                '</div>'));
        });
    });
}