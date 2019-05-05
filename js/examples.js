function setup_inputs(inputs) {
    let clickEvent = new MouseEvent("click", {
        "view": window,
        "bubbles": true,
        "cancelable": false
    });

    let input_button = $('#add-input-button');
    let input_container = $('#autopandas-inputs-container');
    input_container.children().detach();
    input_container.append(input_button);
    for (let i = 0; i < inputs.length; i++) {
        input_button[0].dispatchEvent(clickEvent);
        let inpCodeElem = $('#iocode' + (i + 1));
        inpCodeElem.find('.pycode').each(function () {
            $.data(this, "editor").setValue(inputs[i], 1);
        });
    }
}

function setup_output(output) {
    let outCodeElem = $('#iocode0');
    outCodeElem.find('.pycode').each(function () {
        $.data(this, "editor").setValue(output, 1);
    });
}

function setup_examples() {
    let clickEvent = new MouseEvent("click", {
        "view": window,
        "bubbles": true,
        "cancelable": false
    });

    $('#example-button1').click(() => {
        setup_inputs(["pd.DataFrame({'k1': {0: 'one', 1: 'one', 2: 'one', 3: 'two', 4: 'two', 5: 'two', 6: 'two'}, 'k2': {0: 11, 1: 11, 2: 12, 3: 13, 4: 13, 5: 14, 6: 14}})"]);
        setup_output("pd.DataFrame({'k1': {0: 'one', 2: 'one', 3: 'two', 5: 'two'}, 'k2': {0: 11, 2: 12, 3: 13, 5: 14}})");
    });

    $('#example-button2').click(() => {
        setup_inputs(["pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(10, 17)})",
                             "pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(20, 23)})"]);
        setup_output("pd.DataFrame({'lkey': {0: 'b', 1: 'b', 2: 'b', 3: 'a', 4: 'a', 5: 'a'}, 'data1': {0: 10, 1: 11, 2: 16, 3: 12, 4: 14, 5: 15}, 'rkey': {0: 'b', 1: 'b', 2: 'b', 3: 'a', 4: 'a', 5: 'a'}, 'data2': {0: 21, 1: 21, 2: 21, 3: 20, 4: 20, 5: 20}})")
    });
}