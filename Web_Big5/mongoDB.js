const mongoClient=require("mongodb").MongoClient
const url="mongodb+srv://GreenBig5:1QgvDJucHUpHxEbn@greenbig5.uszbe.mongodb.net/myFirstDatabase?retryWrites=true&w=majority";
const client=new mongoClient(url, { useUnifiedTopology: true}, { useNewUrlParser: true })
module.exports=client;