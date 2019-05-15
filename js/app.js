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

function initialize_clipboard() {
    let clipboard = new ClipboardJS('.copybtn', {
        target: function (trigger) {
            let result = trigger.previousElementSibling;
            $(result).parent().removeClass('noselect');
            return result;
        }
    });


    clipboard.on('success', function(event) {
        let text = event.text;
        $(event.trigger).parent().find("[id*='autopandas-result']").each(function () {
            let idx = $(this).attr("id").substr(17);
            let uid = $(this).data("uid");
            ajax_lambda({task: 'mark_accepted', uid: uid, idx: idx, code: text});
        });

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

        //  First we verify the input-output example.
        //  If it is successful, we get back a uid that we then pass to the engine service
        $('.autopandas-results').show();
        $('#a-autopandas-results').tab('show');

        $(results_container).append(custom_waiting_logo('Validating'));
        let validation_query = ajax_lambda({task: 'validate', inputs: payload.inputs, output: payload.output});
        let engine_query = validation_query.then(
            function (msg) {
                payload.uid = msg.uid;
                return ajax_engine(payload);
            },

            function (error) {
                if (error.responseJSON) {
                    let log = error.responseJSON['log'];
                    warn("Encountered Error : " + log);
                } else {
                    warn("Task validation failed");
                }

                $(results_container).find(':first').remove();
                $(results_container).append($("<p>Could not validate task. Please try again</p>"));
                synthesize_button.show();
                synthesis_cancel_button.hide();
            }
        );

        engine_query.then(
            function (resp) {
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

            function (resp) {
                warn("Could not submit task. Please try again later");
                $(results_container).find(':first').remove();
                $(results_container).append($("<p>Could not submit task. Please try again later</p>"));
                synthesize_button.show();
                synthesis_cancel_button.hide();
                cur_task = null;
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
        let cntr = inputs_container.childElementCount;  // Start from zero
        inputs_container.insertBefore(create_user_input('inps[' + (cntr - 1) + ']', cntr), add_input_button);
    });

    $('.js-tooltip').tooltip();
    $(document).on('click', '.highlight-elem', function(e) {
        defocus_element(e.target);
    });

    initialize_clipboard();
    initialize_synthesizer();
    setup_examples();
    make_tutorial();
}