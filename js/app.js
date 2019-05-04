function create_user_input(caption, id_num) {
    let template = $('#autopandas-default-user-input')[0].children[0];
    let result = template.cloneNode(true);

    //  Append the id_num to each of the ids to ensure uniqueness
    $(result).find("[id]").each(function () {
        this.id = this.id.replace(/\d+$/, "") + id_num;
    });

    //  Similarly fix the hrefs in the preview and code columns
    $(result).find("a[data-toggle][href]").each(function () {
        this.href = $(this).attr('href').replace(/\d+$/, "") + id_num;
    });

    //  Set the caption
    $(result).find("[id*='caption']").each(function () {
        $(this).text(caption);
    });

    //  Setup the preview to a default
    $(result).find('.io-preview').each(function () {
        makePreview(this, 'dataframe', null);
    });

    $(result).find('.pycode').each(function () {
        let preview_link = $(result).find("[id*='a-iopreview']")[0];
        makeUserInputEditor(this, preview_link);
    });

    return result;
}

function initialize_results_container() {
    let clipboard = new ClipboardJS('.copybtn', {
        target: function (trigger) {
            let result = trigger.previousElementSibling;
            $(result).parent().removeClass('noselect');
            return result;
        }
    });


    clipboard.on('success', function(event) {
        event.clearSelection();
        $(event.trigger).parent().addClass('noselect');
        $(event.trigger).attr("data-original-title", "Copied!");
        $(event.trigger).tooltip('show');
        $(event.trigger).attr("data-original-title", "Copy");
        window.setTimeout(function () {
            $(event.trigger).tooltip('hide');
        }, 2000);
    });

    clipboard.on('error', function(event) {
        $(event.trigger).attr("data-original-title", "Something went wrong...");
        $(event.trigger).tooltip('show');
        $(event.trigger).attr("data-original-title", "Copy");
        window.setTimeout(function () {
            $(event.trigger).tooltip('hide');
        }, 2000);
    });
}

function add_result(code) {
    let results_container = $('#div-autopandas-result-codes')[0];
    let result = $('' +
        '<pre id="result" class="copytoclipboard noselect">' +
        '<code class=" language-python">' + code + '</code>' +
        '<button type="button" class="btn btn-default copybtn js-tooltip" data-toggle="tooltip" data-placement="top" title="Copy">' +
        '<i class="far fa-copy fa-lg"></i>' +
        '</button></pre>');
    let id_num = results_container.childElementCount;

    //  Append the id_num to each of the ids to ensure uniqueness
    result.find("[id]").each(function () {
        this.id = this.id.replace(/\d+$/, "") + id_num;
    });

    result.find(".copybtn").each(function () {
        $(this).tooltip();
    });

    $(results_container).append(result);
    Prism.highlightAll();
}

function initialize_synthesizer() {
    let cur_task = null;
    let synthesize_button = $('#synthesize-button');
    let synthesis_cancel_button = $('#synthesize-cancel-button');
    synthesize_button.click(() => {
        synthesize_button.hide();
        synthesis_cancel_button.show();

        let payload = create_synthesis_task();
        let results_container = $('#div-autopandas-result-codes')[0];
        $(results_container).empty();
        $(results_container).append(custom_waiting_logo('Initializing'));

        $('.divider').show();
        $('.autopandas-results').show();
        $.ajax(
            {
                type: "POST",
                url: "http://127.0.0.1:5000/autopandas",
                data: JSON.stringify(payload),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
                
                success: function (resp) {
                    let uid = resp.uid;
                    //  Instantiate a poller
                    let poller = solution_poller(uid, results_container);

                    let timer = window.setTimeout(poller, 2000);
                    cur_task = {
                        uid: uid,
                        poller: poller,
                        timer: timer
                    };
                    success_alert("Task Successfully Submitted");
                },

                error: function (resp) {
                    warn("Something went wrong. Try again");
                    synthesize_button.show();
                    synthesis_cancel_button.hide();
                    cur_task = null;
                }
            }
        );
    });

    synthesis_cancel_button.click(() => {
        if (cur_task != null) {
            window.clearTimeout(cur_task.timer);
            cur_task = null;
        }

        synthesis_cancel_button.hide();
        synthesize_button.show();
    });
}

function initialize() {
    //  Setup the output.
    let output_container = $('#autopandas-output-container')[0];
    $(output_container).append(create_user_input('', 0));

    //  Setup the inputs. By default there are no inputs, just the button
    let add_input_button = $("<button id=\"add-input-button\" class='btn btn-danger'>Add Input</button>")[0];
    let inputs_container = $('#autopandas-inputs-container')[0];
    $(inputs_container).append(add_input_button);
    $(add_input_button).click(() => {
        let cntr = inputs_container.childElementCount;
        inputs_container.insertBefore(create_user_input('Input-' + cntr, cntr), add_input_button);
    });

    $('.js-tooltip').tooltip();
    initialize_results_container();
    initialize_synthesizer();
}