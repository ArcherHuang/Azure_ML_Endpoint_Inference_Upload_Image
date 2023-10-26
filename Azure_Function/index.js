const multipart = require("parse-multipart");
const FormData = require('form-data');
const { BlobServiceClient } = require('@azure/storage-blob');

BLOB_CONNECTION_STRING = ""
BLOB_CONTAINER_NAME = "images"

module.exports = async function (context, req) {
    context.log('JavaScript HTTP trigger function processed a request.');
    
    var bodyBuffer = Buffer.from(req.body);
    var boundary = multipart.getBoundary(req.headers['content-type']);
    var parts = multipart.Parse(bodyBuffer, boundary);

    try {
        const currentTimestamp = new Date().getTime();
        context.log(`currentTimestamp: ${currentTimestamp}`);
        context.log(`type: ${parts[0].type}`);
        let subName = parts[0].type.replace("image/", "");

        const blobServiceClient = await BlobServiceClient.fromConnectionString(BLOB_CONNECTION_STRING);
        const containerClient = await blobServiceClient.getContainerClient(BLOB_CONTAINER_NAME);
        const blockBlobClient = containerClient.getBlockBlobClient(`${currentTimestamp}.${subName}`);
        const uploadBlobResponse = await blockBlobClient.upload(parts[0].data, parts[0].data.length);
        context.log(`uploadBlobResponse: ${JSON.stringify(uploadBlobResponse)}`);

        const formData = new FormData();
        formData.append('image', parts[0].data);
        const response = await fetch('https://NAME.LOCATION.inference.ml.azure.com/score', {
          method: 'POST',
          headers: {
            Authorization: 'Bearer KEY',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            fileUrl: `https://NAME.blob.core.windows.net/images/${currentTimestamp}.${subName}`,
          }),
        });

        context.log(`response: ${JSON.stringify(response)}`);
        context.res = { 
            body : {
                from: 'Azure Function',
                name : parts[0].filename,
                type: parts[0].type,
                length: parts[0].data.length,
                fileName: currentTimestamp,
                subName,
                result: response,
            }
        };
    } catch (error) {
        context.log(error);
        context.res = {
            status: 500,
            message: 'Error calling the external API: ' + error,
        };
    }
}