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

    $('#synthesize-button').click(() => {
        let results_container = $('#div-autopandas-result-codes')[0];
        $(results_container).empty();
        let searching_logo = custom_waiting_logo('Searching...');
        $(results_container).append(searching_logo);

        $('.divider').show();
        $('.autopandas-results').show();
        window.setTimeout(function () {
            add_result("output = inps[0].some_magical_function(some_magical_arguments)");
        }, 5000);

        window.setTimeout(function () {
            searching_logo.remove();
        }, 10000);


        $.ajax(
            {
                type: "POST",
                url: 'http://127.0.0.1:5000/synthesis',
                data: JSON.stringify({
                    task: 'synthesis',
                    inputs: ['pd.DataFrame({\'k1\': {0: \'one\', 1: \'one\', 2: \'one\', 3: \'two\', 4: \'two\', 5: \'two\', 6: \'two\'}, \'k2\': {0: 11, 1: 11, 2: 12, 3: 13, 4: 13, 5: 14, 6: 14}})'],
                    output: 'pd.DataFrame({\'k1\': {0: \'one\', 2: \'one\', 3: \'two\', 5: \'two\'}, \'k2\': {0: 11, 2: 12, 3: 13, 5: 14}})',
                }),
                contentType: "application/json; charset=utf-8",
                dataType: "json",

                success: function (msg) {
                    console.log(msg);
                },

                error: function (errormessage) {
                    console.log(errormessage);
                }
            }
        );
    });
}