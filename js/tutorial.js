function make_tutorial() {
    //  Refactor later
    $('#interactive-tutorial').click(() => {


        setup_output("df = pd.DataFrame({'beer_servings': {'Africa': 121.0, 'Asia': 0.1, 'Europe': 167.0}, 'spirit_servings': {'Africa': 28.75, 'Asia': 0.2, 'Europe': 135.0}, 'wine_servings': {'Africa': 29.5, 'Asia': 0.3, 'Europe': 183.0}, 'total_litres_of_pure_alcohol': {'Africa': 3.3, 'Asia': 0.6, 'Europe': 8.65}})\n" +
                "df.index.names = ['continent']\n" +
                "df");
        let add_input_button = $('#add-input-button')[0];
        tooltip_message(add_input_button, "Click here to add inputs to the input-output example", 10000);
        focus_element(add_input_button);
        $(add_input_button).one('click', function () {
            hide_tooltip(add_input_button);
            setup_inputs(["pd.DataFrame(\n" +
            " {'country': {0: 'Afghanistan', 1: 'Albania', 2: 'Algeria', 3: 'Andorra', 4: 'Angola'},\n" +
            "  'beer_servings': {0: 0.1, 1: 89, 2: 25, 3: 245, 4: 217},\n" +
            "  'spirit_servings': {0: 0.2, 1: 132, 2: 0.5, 3: 138, 4: 57},\n" +
            "  'wine_servings': {0: 0.3, 1: 54, 2: 14, 3: 312, 4: 45},\n" +
            "  'total_litres_of_pure_alcohol': {0: 0.6, 1: 4.9, 2: 0.7, 3: 12.4, 4: 5.9},\n" +
            "  'continent': {0: 'Asia', 1: 'Europe', 2: 'Africa', 3: 'Europe', 4: 'Africa'}})"]);

            let preview_button = $('#a-iopreview1')[0];
            tooltip_message(preview_button, "Click preview to execute the code and obtain the result. " +
                "You can edit the preview directly and see the changes reflected in the code", 10000);
            focus_element(preview_button);

            $(preview_button).one('click', function () {
                hide_tooltip(preview_button);
                let output_preview_button = $('#a-iopreview0')[0];
                tooltip_message(output_preview_button, "Similarly click the preview to visualize the target output", 10000);

                focus_element(output_preview_button);
                $(output_preview_button).one('click', function () {
                    hide_tooltip(output_preview_button);
                    let synthesize_button = $('#synthesize-button')[0];
                    tooltip_message(synthesize_button, "Click Synthesize to run the engine", 10000);
                    focus_element(synthesize_button);

                    $(synthesize_button).one('click', function () {
                        hide_tooltip(synthesize_button);
                        let results_tab = $('#a-autopandas-results')[0];
                        tooltip_message(results_tab, "Synthesized programs will be displayed here", 10000);
                    });
                });
            });
        });
    });
}