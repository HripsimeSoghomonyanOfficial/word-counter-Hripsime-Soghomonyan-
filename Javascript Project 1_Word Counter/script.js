document.getElementById('countButton').addEventListener('click', function() {
    var text = document.getElementById('textInput').value;
    var letterCount = text.length;
    var wordCount = text.trim().split(/\s+/).length;
    var sentenceCount = text.split(/[.!?]+/).length - 1;
    
    document.getElementById('letterCount').textContent = 'Letters: ' + letterCount;
    document.getElementById('wordCount').textContent = 'Words: ' + wordCount;
    document.getElementById('sentenceCount').textContent = 'Sentences: ' + sentenceCount;
});

document.getElementById('countButton').addEventListener('click', function() {
    var randomColor = '#' + Math.floor(Math.random()*16777215).toString(16);
    document.body.style.backgroundColor = randomColor;
});