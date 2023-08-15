const tfjsNode = require("@tensorflow/tfjs-node");

const testData = [
  "The movie was great, I really liked it!",
  "The movie was okay. Not really that great, could be better",
  "The movie was terrible...",
  "The movie was bad, I really hated it! Complete disaster",
];

async function main() {
  console.log("Loading model...");
  const model = await tfjsNode.node.loadSavedModel(
    "../../models/imdb_reviews_model.tf"
  );

  // This is as of now (August 2023) not yet supported
  //model.execute({ input: testData }).print();

  const output = model.predict(tfjsNode.tensor(testData));
  console.log("output:", output.arraySync());
}
main();
