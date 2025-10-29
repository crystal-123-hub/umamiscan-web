# app.py
from flask import Flask, request, jsonify, render_template_string
import os
from predict import run_prediction
import torch
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>UmamiScan</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }
        .container { display: flex; gap: 20px; margin-top: 20px; }
        .box { 
            flex: 1; 
            border: 1px solid #ddd; 
            padding: 15px; 
            border-radius: 8px; 
            background: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        textarea { 
            width: 100%; 
            height: 300px; 
            padding: 10px; 
            border: 1px solid #ccc; 
            border-radius: 5px; 
            font-family: monospace; 
            resize: vertical;
        }
        button { 
            padding: 10px 20px; 
            background: #e63946; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            margin-top: 10px;
            font-size: 14px;
        }
        button:hover { background: #c1121f; }
        .output { 
            background: #f0f0f0; 
            padding: 10px; 
            border-radius: 5px; 
            max-height: 300px; 
            overflow-y: auto; 
            font-family: monospace;
            white-space: pre-wrap;
        }
        h1 { color: #1d3557; }
        p { color: #555; }
    </style>
</head>
<body>
    <h1>UmamiScan</h1>
    <p>Predict umami probability of peptides using deep learning (ESM + Graph).</p>
    <p>Contact: <a href="mailto:1776228595@qq.com">1776228595@qq.com</a></p>

    <div class="container">
        <div class="box">
            <h2>Input FASTA</h2>
            <textarea id="input" placeholder="&gt;Pep1&#10;SHELDSASSEVN&#10;&gt;Pep2&#10;VPVQA"></textarea>
            <button onclick="submit()">Analyze with UmamiScan</button>
        </div>
        <div class="box">
            <h2>Results</h2>
            <div id="output" class="output">Waiting for input...</div>
            <button onclick="downloadCSV()">Download CSV</button>
            <button onclick="downloadTXT()">Download TXT</button>
        </div>
    </div>

    <script>
        let resultText = "";
        const uuid = Date.now(); // 简单区分不同用户的请求

        function submit() {
            const input = document.getElementById('input').value.trim();
            if (!input) {
                alert("Please input FASTA format sequences.");
                return;
            }
            fetch('/umamiscan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fasta: input })
            })
            .then(res => res.json())
            .then(data => {
                resultText = data.text_result;
                document.getElementById('output').innerHTML = '<pre>' + resultText + '</pre>';
            })
            .catch(err => {
                document.getElementById('output').innerHTML = '<span style="color:red">Error: ' + err.message + '</span>';
            });
        }

        function downloadCSV() {
            if (!resultText) return alert("No results.");
            const blob = new Blob(["ID,Sequence,Prediction,Probability\\n", resultText], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `umamiscan_results_${uuid}.csv`;
            a.click();
        }

        function downloadTXT() {
            if (!resultText) return alert("No results.");
            const blob = new Blob([resultText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `umamiscan_results_${uuid}.txt`;
            a.click();
        }
    </script>
</body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/umamiscan', methods=['POST'])
def predict_route():
    data = request.json
    fasta_text = data.get('fasta', '').strip()

    if not fasta_text:
        return jsonify({"error": "Empty input"}), 400

    try:
        # 使用唯一文件名避免冲突
        output_file = f"results/results_{int(torch.randn(1).item() * 1e6)}.csv"
        result_text, saved_file = run_prediction(fasta_text, output_file)
        return jsonify({"text_result": result_text})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)