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

function setup_example($elem, inputs, output) {
    $elem.click(() => {
        setup_inputs(inputs);
        setup_output(output);

        //$('[id*="a-iopreview"]').each(function () {
            //fire_click(this);
        //});

        $('[id*="a-iocode"]').each(function () {
            fire_click(this);
        });

        tooltip_message($('#synthesize-button')[0], 'Click Synthesize to run the engine!', 5000);
    });

}

function setup_examples() {
    let clickEvent = new MouseEvent("click", {
        "view": window,
        "bubbles": true,
        "cancelable": false
    });

    setup_example($('#example-button3'),
        ["pd.DataFrame({'k1': {0: 'one', 1: 'one', 2: 'one', 3: 'two', 4: 'two', 5: 'two', 6: 'two'}, 'k2': {0: 11, 1: 11, 2: 12, 3: 13, 4: 13, 5: 14, 6: 14}})"],
        "pd.DataFrame({'k1': {0: 'one', 2: 'one', 3: 'two', 5: 'two'}, 'k2': {0: 11, 2: 12, 3: 13, 5: 14}})");

    setup_example($('#example-button2'),
        ["pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(10, 17)})",
                             "pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(20, 23)})"],
        "pd.DataFrame({'lkey': {0: 'b', 1: 'b', 2: 'b', 3: 'a', 4: 'a', 5: 'a'}, 'data1': {0: 10, 1: 11, 2: 16, 3: 12, 4: 14, 5: 15}, 'rkey': {0: 'b', 1: 'b', 2: 'b', 3: 'a', 4: 'a', 5: 'a'}, 'data2': {0: 21, 1: 21, 2: 21, 3: 20, 4: 20, 5: 20}})");

    setup_example($('#example-button1'),
        ["pd.DataFrame(\n" +
        " {'country': {0: 'Afghanistan',\n" +
        "              1: 'Albania',\n" +
        "              2: 'Algeria',\n" +
        "              3: 'Andorra',\n" +
        "              4: 'Angola'},\n" +
        "  'beer_servings': {0: 0.1, 1: 89, 2: 25, 3: 245, 4: 217},\n" +
        "  'spirit_servings': {0: 0.2, 1: 132, 2: 0.5, 3: 138, 4: 57},\n" +
        "  'wine_servings': {0: 0.3, 1: 54, 2: 14, 3: 312, 4: 45},\n" +
        "  'total_litres_of_pure_alcohol': {0: 0.6, 1: 4.9, 2: 0.7, 3: 12.4, 4: 5.9},\n" +
        "  'continent': {0: 'Asia', 1: 'Europe', 2: 'Africa', 3: 'Europe', 4: 'Africa'}}\n" +
        ")"],
        "df = pd.DataFrame({'beer_servings': {'Africa': 121.0, 'Asia': 0.1, 'Europe': 167.0}, 'spirit_servings': {'Africa': 28.75, 'Asia': 0.2, 'Europe': 135.0}, 'wine_servings': {'Africa': 29.5, 'Asia': 0.3, 'Europe': 183.0}, 'total_litres_of_pure_alcohol': {'Africa': 3.3, 'Asia': 0.6, 'Europe': 8.65}})\n" +
            "df.index.names = ['continent']\n" +
            "df");
}
