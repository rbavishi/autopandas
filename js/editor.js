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
        let baseUrl = "https://dz4k5ce9l1.execute-api.us-east-2.amazonaws.com";
        let stage = "default";
        let interactionService = baseUrl + "/" + stage + "/AutoPandasInteractionEngine";

        //  Adding a loading sign for clarity
        let parentId = $(pycode).parent().parent().attr('id');
        let previewElem = $('#iopreview' + parentId.substr(-1));
        previewElem.empty();
        previewElem.append(custom_waiting_logo('Loading'));
        $.ajax(
            {
                type: "POST",
                url: interactionService,
                data: JSON.stringify({task: 'evalcode', input: code}),
                contentType: "application/json; charset=utf-8",
                dataType: "json",

                success: function (msg) {
                    makePreview(previewElem[0], msg.dtype, msg.data);
                    let previewElemList = $('#a-iopreview' + parentId.substr(-1));
                    previewElemList.tab('show');
                },

                error: function (errormessage) {
                    let log = "";
                    if (errormessage.responseJSON) {
                        log = "Encountered Error : " + errormessage.responseJSON['log'];
                    } else {
                        log = "Something went wrong. Check your code";
                    }

                    warn(log);
                    previewElem.empty();
                    previewElem.append($('<div class="alert alert-dismissible alert-danger">' +
                        '<p>' + log + '</p>' +
                        '</div>'));
                }
            }
        );
    });
}