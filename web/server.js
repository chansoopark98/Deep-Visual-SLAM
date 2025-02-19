var express = require('express');
var app = express();
var cors = require('cors');

let fs = require('fs');
let options = {
    key: fs.readFileSync('./tsp-xr.com_20241010671FC.key.pem'),
    cert: fs.readFileSync('./tsp-xr.com_20241010671FC.crt.pem'),
    requestCert: false,
    rejectUnauthorized: false
};
app.set('view engine', 'ejs');
app.engine('html', require('ejs').renderFile);

app.use(cors());
console.log(__dirname);
app.use('/assets', express.static(__dirname + '/assets'));
app.use('/styles', express.static(__dirname + '/styles'));
app.use('/modules', express.static(__dirname + '/modules'));
app.use('/build', express.static(__dirname + '/node_modules/three/build'));
app.use('/gltf', express.static(__dirname + '/node_modules/three/'));
var server_port = 5555;
var server = require('https').createServer(options, app);

app.get('/', (req, res) => {
  
    res.render(__dirname + "/cam.html");    // index.ejs을 사용자에게 전달
})

server.listen(server_port, function() {
  console.log( 'Express server listening on port ' + server.address().port );
});