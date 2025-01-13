import streamlit as st
import neuron

st.title("Simulador de neurona")
st.image("https://www.psycolab.com/wp-content/uploads/15286-scaled.jpg", width=600)

numero_pesos_entradas = st.slider("Elige el número de entradas/pesos que tendrá la neurona:", min_value=1, max_value=10, step=1)

st.title("Pesos")
pesos = []
for i in range(int(numero_pesos_entradas)):
    pesos.append(st.number_input(f"W{i+1}"))
st.text(f"W = {pesos}")

st.title("Entradas")
entradas = []
for i in range(int(numero_pesos_entradas)):
    entradas.append(st.number_input(f"X{i+1}"))
st.text(f"x = {entradas}")

st.title("Sesgo")
sesgo = st.number_input("Introduce el valor del sesgo")

st.title("Función de activación")
funcion_activacion = st.selectbox("Elige la función de activación", ["Sigmoide", "ReLU", "Tangente hiperbólica", "Binary Step"])

if st.button("Calcular la salida"):
    n1 = neuron.Neuron(entradas, pesos, sesgo, funcion_activacion)
    output = n1.predict()
    st.write(f"La salida de la neurona es: {output}")
