import React, { useState } from 'react';

function App() {
  const [cholesterol, setCholesterol] = useState(0);
  const [glucose, setGlucose] = useState(0);
  const [hdlChol, setHdlChol] = useState(0);
  const [cholHdlRatio, setCholHdlRatio] = useState(0);
  const [age, setAge] = useState(0);
  const [gender, setGender] = useState(0); // Puedes tener opciones predefinidas si es un campo de selección
  const [height, setHeight] = useState(0);
  const [weight, setWeight] = useState(0);
  const [bmi, setBmi] = useState(0);
  const [systolicBP, setSystolicBP] = useState(0);
  const [diastolicBP, setDiastolicBP] = useState(0);
  const [waist, setWaist] = useState(0);
  const [hip, setHip] = useState(0);
  const [waistHipRatio, setWaistHipRatio] = useState(0);
  const [result, setResult] = useState(null);


  const handleSubmit = async (e) => {
    e.preventDefault();

    try{
    const response = await fetch('http://127.0.0.1:5000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        cholesterol,
        glucose,
        hdlChol,
        cholHdlRatio,
        age,
        gender,
        height,
        weight,
        bmi,
        systolicBP,
        diastolicBP,
        waist,
        hip,
        waistHipRatio,
      }),
    });

    const result = await response.json();
    console.log('Predicción:', result.prediction);
    setResult(result.prediction);
  }catch(error){
    console.error('Error al realizar la predicción', error);
  }
  };

  return (
    <div className="App">
      <h1>Predicciones de Diabetes</h1>
      <form onSubmit={handleSubmit}>
        {/* ... (otros campos) */}
        <label>
          1. Colesterol:
          <input type="number" value={cholesterol} onChange={(e) => setCholesterol(e.target.value)} />
        </label>
        <br />
        <label>
          2. Glucosa:
          <input type="number" value={glucose} onChange={(e) => setGlucose(e.target.value)} />
        </label>
        <br />
        <label>
          3. HDLChol:
          <input type="number" value={hdlChol} onChange={(e) => setHdlChol(e.target.value)} />
        </label>
        <br />
        <label>
          4. CholHDLRatio:
          <input type="number" value={cholHdlRatio} onChange={(e) => setCholHdlRatio(e.target.value)} />
        </label>
        <br />
        <label>
          5. Age:
          <input type="number" value={age} onChange={(e) => setAge(e.target.value)} />
        </label>
        <br />
        <label>
          6. Gender:
          <input type="number" value={gender} onChange={(e) => setGender(e.target.value)} />
        </label>
        <br />
        <label>
          7. Height:
          <input type="number" value={height} onChange={(e) => setHeight(e.target.value)} />
        </label>
        <br />
        <label>
          8. Weight:
          <input type="number" value={weight} onChange={(e) => setWeight(e.target.value)} />
        </label>
        <br />
        <label>
          9. BMI:
          <input type="number" value={bmi} onChange={(e) => setBmi(e.target.value)} />
        </label>
        <br />
        <label>
          10. SystolicBP:
          <input type="number" value={systolicBP} onChange={(e) => setSystolicBP(e.target.value)} />
        </label>
        <br />
        <label>
          11. DiastolicBP:
          <input type="number" value={diastolicBP} onChange={(e) => setDiastolicBP(e.target.value)} />
        </label>
        <br />
        <label>
          12. Waist:
          <input type="number" value={waist} onChange={(e) => setWaist(e.target.value)} />
        </label>
        <br />
        <label>
          13. Hip:
          <input type="number" value={hip} onChange={(e) => setHip(e.target.value)} />
        </label>
        <br />
        <label>
          14. WaistHipRatio:
          <input type="number" value={waistHipRatio} onChange={(e) => setWaistHipRatio(e.target.value)} />
        </label>
        <br />
        
        <button type="submit">Realizar predicción</button>
      </form>

    {/* Mostrar resultados si están disponibles */}
      {result !== null && (
        <div>
          <h2>Resultado de la Predicción:</h2>
          <p>{result}</p>
        </div>
      )}

    </div>
  );
}

export default App;
