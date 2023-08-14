const tfjsNode = require("@tensorflow/tfjs-node");
const tfjs = require("@tensorflow/tfjs");

async function main() {
  console.log("Loading model...");
  // const backend = await tfjsNode.setBackend("tensorflow");
  // console.log("Backend setup succesfully:", backend);
  // await tfjsNode.backend();
  // await tfjsNode.ready();
  const model = await tfjsNode.node.loadSavedModel(
    "../models/imdb_reviews_model.tf",
    ["serve"],
    "serving_default"
  );
  const testData = [
    "The movie was great, I really liked it!",
    "The movie was okay. Not really that great, could be better",
    "The movie was terrible...",
    "The movie was bad, I really hated it! Complete disaster",
  ];
  console.log(model.inputs);
  //model.execute({ input: testData }).print();
  const output = model.predict(tfjsNode.tensor(testData));
  console.log("output:", output.arraySync());
}
main();
