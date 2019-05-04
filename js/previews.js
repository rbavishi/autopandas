function makeDataFramePreview(hostDiv, data) {
    if (data == null) {
        data = [['', 'C1', 'C2'],
                ['0', 'E1', 'E2'],
                ['1', 'E3', 'E4']]
    }

    let tableDiv = $("<div class='hot handsontable htColumnHeaders'></div>");
    let saveChangesButton = $("<button class='btn btn-outline-info btn-sm'>Save Changes</button>");
    $(hostDiv).empty();
    $(hostDiv).append(tableDiv);
    $(hostDiv).append(saveChangesButton);

    let hot = new Handsontable(tableDiv[0], {
        data: data,
        contextMenu: true,
        licenseKey: 'non-commercial-and-evaluation'
    });

    let exportPlugin = hot.getPlugin('exportFile');
    saveChangesButton.click(() => {
        let string_repr = exportPlugin.exportAsString('csv', {
            rowDelimiter: '\n',
            bom: false
        });
        let baseUrl = "https://dz4k5ce9l1.execute-api.us-east-2.amazonaws.com";
        let stage = "default";
        let interactionService = baseUrl + "/" + stage + "/AutoPandasInteractionEngine";
        $.ajax(
            {
                type: "POST",
                url: interactionService,
                data: JSON.stringify({task: 'tocode', dtype:'dataframe', val: string_repr}),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
                success: function (msg) {
                    console.log(msg.code);
                    let parentId = $(hostDiv).attr('id');
                    console.log(parentId);
                    let codeElem = $('#iocode' + parentId.substr(-1));
                    codeElem.find('.pycode').each(function () {
                        $.data(this, "editor").setValue(msg.code, 1);
                    });
                    let codeElemList = $('#a-iocode' + parentId.substr(-1));
                    codeElemList.tab('show');
                },
                error: function (errormessage) {
                    if (errormessage.responseJSON) {
                        let log = errormessage.responseJSON['log'];
                        warn("Encountered Error : " + log);
                    } else {
                        warn("Something went wrong.");
                    }
                }
            }
        );
    });
}

function makePreview(hostDiv, dtype, data) {
    if (dtype == null) dtype = 'dataframe';

    if (dtype === 'dataframe') {
        makeDataFramePreview(hostDiv, data);
    }

}