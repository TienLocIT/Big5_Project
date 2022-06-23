const express=require("express")
const app=express()
const port=process.env.PORT||5000;
const auth=require("./middleware/auth.admin");
var session = require('express-session');
const client=require("./mongoDB")
const cors=require("cors");
const expertRouter = require("./router/expert.router");

const userRouter=require("./router/user.router");
const adminRouter=require("./router/admin.router");
const cookieParser = require("cookie-parser");
app.use(express.static("public"));
app.set("view engine","ejs");
app.set("views","./views");
app.use(cors());
app.use(express.json()) // for parsing application/json
app.use(express.urlencoded({ extended: true }))
app.use(cookieParser("fnfnhjkfhjkhasjk"));
app.use(session({
    resave: true, 
    saveUninitialized: true, 
    secret: 'addQuestions', 
    cookie: { maxAge: 60000 }}));
require("dotenv").config();
app.get("/",async (req,res)=>{
    //  res.setHeader("Content-Type", "text/html");
    const dbs=await client.connect().catch(err=>{
        console.log(err);
    })
    const db=await dbs.db("green_big_5");
    if(!req.signedCookies.admin_id){
        // res.setHeader("Content-Type", "text/html");
        return res.redirect("/login");
    }
    const admin=await db.collection("admin").findOne({"id":req.signedCookies.admin_id});
    if(admin){
        // res.setHeader("Content-Type", "text/html");
        return res.redirect("/admin")
    }
    else{
        // res.setHeader("Content-Type", "text/html");
        return res.redirect("/login");
    }

})
app.get("/login",(req,res)=>{
    // res.setHeader("Content-Type", "text/html");
    res.render("pages/login")
})
app.post("/login",async(req,res)=>{
    const username=req.body.username;
    const password=req.body.password;
      const dbs=await client.connect().catch(err=>{
        console.log(err);
    })
    const db=await dbs.db("green_big_5");
    const dbWithUserName=await db.collection("admin").findOne({"username":username});
    if(dbWithUserName){
        const dbWithUserNamePass=await db.collection("admin").findOne({"username":username,"password":password});
        if(dbWithUserNamePass){
            res.cookie("admin_id",dbWithUserNamePass.id,{
                signed:true
            });
            return res.status(200).json({"error":"No"})
        }
        else{
            return res.status(200).json({"error":"Password is not correct"});
        }
    }
    else{
        return res.status(200).json({"error":"Username is not correct"});
    }
})
// app.get("/listQuestion",controller.listQuestion);

// app.post("/user",async (req,res)=>{
//     const user=await db.collection("users").findOne({username:req.body.username})
//     if(user){
//         res.status(500).json(user)
//     }
//     else{
//         res.status(200).json(null);
//     }
        
// })
// app.get("/questions",async (req,res)=>{
//     var id=req.query.id;
//     const user=await db.collection("users").findOne({_id:ObjectId(id)});
//     if(user){
//         const question=await db.collection("questions").find({}).toArray();
//         res.status(500).json(question);
//     }
//     else{
//         res.send("Wrong");
//     }
    
// })
// app.post("/questions",async (req,res)=>{
//         var id=req.query.id;
//         const user=await db.collection("users").findOne({_id:ObjectId(id)});
//         if(user){
//             await db.collection("users").updateOne({_id:ObjectId(id)},{
//                 $set:{"answer":req.body}},{multi:true}).catch(error => console.error(error))
//                 res.status(500).json(user)

//         }
//         else{
//             res.send("Wrong")
//         }
        
              
// })


app.use("/users",userRouter);
app.use("/admin",adminRouter);
app.use("/expert",expertRouter);
app.listen(port,()=>{
    console.log("App listening at:"+port)
})

body = {
    "gender":1,
    "age":2,
    "location":2
}