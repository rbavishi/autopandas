function makeDataFramePreview(hostDiv, data) {
    if (data == null) {
        data = [['', 'C1', 'C2'],
                ['0', 'E1', 'E2'],
                ['1', 'E3', 'E4']]
    }

    let tableDiv = $("<div class='hot handsontable htColumnHeaders'></div>");
    let saveChangesButton = $("<button class='btn btn-outline-info btn-sm'>Save Changes</button>");
    saveChangesButton.hide();
    $(hostDiv).empty();
    $(hostDiv).append(tableDiv);
    $(hostDiv).append(saveChangesButton);

    let changeObserved = false;
    let hot = new Handsontable(tableDiv[0], {
        data: data,
        contextMenu: true,
        afterChange: (changes, source) => {
            if (source !== 'loadData' && !changeObserved) {
                changeObserved = true;
                saveChangesButton.show();
            }
        },
        licenseKey: 'non-commercial-and-evaluation'
    });

    let exportPlugin = hot.getPlugin('exportFile');
    saveChangesButton.click(() => {
        let string_repr = exportPlugin.exportAsString('csv', {
            rowDelimiter: '\n',
            bom: false
        });

        let tocode_query = ajax_lambda({task: 'tocode', dtype: 'dataframe', val: string_repr});
        tocode_query.done(function (msg) {
            console.log(msg.code);
            let parentId = $(hostDiv).attr('id');
            console.log(parentId);
            let codeElem = $('#iocode' + parentId.substr(-1));
            codeElem.find('.pycode').each(function () {
                $.data(this, "editor").setValue(msg.code, 1);
            });
            let codeElemList = $('#a-iocode' + parentId.substr(-1));
            codeElemList.tab('show');

        }).fail(function (error) {
            if (error.responseJSON) {
                let log = error.responseJSON['log'];
                warn("Encountered Error : " + log);
            } else {
                warn("Something went wrong.");
            }
        });
    });
}

function makeDefaultPreview(hostDiv, text) {
    $(hostDiv).empty();
    $(hostDiv).append($('<pre>' + text + '</pre>'));
}

function makePreview(hostDiv, dtype, data) {
    if (dtype == null) dtype = 'dataframe';

    if (dtype === 'dataframe') {
        makeDataFramePreview(hostDiv, data);
    } else {
        makeDefaultPreview(hostDiv, data)
    }

}