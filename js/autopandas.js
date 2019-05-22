function create_synthesis_task() {
    let inputs = [];
    let output = [];
    $("#autopandas-inputs-container").find('.pycode').each(function () {
        inputs.push($.data(this, "editor").getValue());
    });
    $("#autopandas-output-container").find('.pycode').each(function () {
        output.push($.data(this, "editor").getValue());
    });
    output = output[0];

    return {
        task: 'synthesis',
        inputs: inputs,
        output: output,
    }
}

function add_result(uid, code, id) {
    let results_container = $('#div-autopandas-result-codes')[0];
    let result = $('' +
        '<p><b>Solution ' + id + ':</b></p><pre class="copytoclipboard noselect line-numbers">' +
        '<code class=" language-python">' + code + '</code>' +
        '<button id="autopandas-result" type="button" class="btn btn-default copybtn js-tooltip" data-toggle="tooltip" data-placement="top" title="Copy">' +
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

    result.find("[id*='autopandas-result']").each(function () {
        $(this).data("uid", uid);
    });

    $(results_container).append(result);
    Prism.highlightAll();
}

function solution_poller(uid, results_container) {
    let prev_solutions = [];
    function poller() {
        let poll_query = ajax_engine({task: 'poll', uid: uid});
        poll_query.done(function (msg) {
            if (msg.status === 'waiting') {
                $(results_container).find(':first').remove();
                $(results_container).prepend(custom_waiting_logo('Waiting in queue'));
                window.setTimeout(poller, 2000);

            } else if (msg.status === 'running') {
                $(results_container).find(':first').remove();
                $(results_container).prepend(custom_waiting_logo('Running'));
                if (msg.solutions && msg.solutions.length > prev_solutions.length) {
                    for(let i = prev_solutions.length; i < msg.solutions.length; i++) {
                        add_result(uid, msg.solutions[i], i + 1);
                    }
                    prev_solutions = msg.solutions;
                }

                window.setTimeout(poller, 2000);

            } else {
                $(results_container).find(':first').remove();
                $('#synthesize-button').show();
                $('#synthesize-cancel-button').hide();
            }
        }).fail(function (error) {
            $(results_container).find(':first').remove();
            $('#synthesize-button').show();
            $('#synthesize-cancel-button').hide();
        });
    }

    return poller;
}