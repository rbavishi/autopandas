function ajax_lambda(payload) {
    let baseUrl = "https://dz4k5ce9l1.execute-api.us-east-2.amazonaws.com";
    let stage = "default";
    let interactionService = baseUrl + "/" + stage + "/AutoPandasInteractionEngine";

    return $.ajax({
            type: "POST",
            url: interactionService,
            data: JSON.stringify(payload),
            contentType: "application/json; charset=utf-8",
            dataType: "json"
    });
}

function ajax_engine(payload) {
    return $.ajax({
        type: "POST",
        url: "https://95btzrmsmb.execute-api.us-east-2.amazonaws.com/basic",
        data: JSON.stringify(payload),
        contentType: "application/json; charset=utf-8",
        dataType: "json"
    });
}
