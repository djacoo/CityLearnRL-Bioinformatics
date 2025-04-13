# CityLearnRL-Bioinformatics
Questo repository raccoglie il codice e i risultati sviluppati durante il lavoro di tesi sull'utilizzo di algoritmi di Reinforcement Learning (RL) per l'ottimizzazione energetica urbana nel simulatore **CityLearn**

**OBIETTIVO DEL PROGETTO**
L'obiettivo principale della tesi é partire dal file _tutorial.ipynb_ del repository ufficiale CityLearn, basato sull'argomento **SAC** (Soft Actor Critic), e **estenderlo con altri algoritmi avanzati** come:
* **PPO** (Proximal Policy Optimization)
* **TD3** (Twim Delayed DDPG)
L'analisi si concentra sul confronto delle performance tra gli algoritmi in termini di reward, stabilitá e rapiditá di apprendimento.

**STRUTTURA DEL REPOSITORY**
- _tutorial.ipynb_ : original file / repo
- _tutorial_ppo.ipynb_ : PPO implemented
- _tutorial_td3.ipynb_ : TD3 implemented
- _results_/ : Grafici e risultati degli esperimenti con diversi seed
- README.md : Questo file

**REQUISITI**
* Python 3.8 (stable) +
* stable-baselines3
* matplotlib
* numpy
* citylearn (versione compatibile con la versione python utilizzata

**RISULTATI ATTESI**
Un buon algoritmo di RL dovrebbe mostrare: 
- Una curva di reward crescente, con basso rumore tra i seed
- Stabilitá tra le esecuzioni
- Apprendimento rapido (meno episodi per convergere


**AUTORE**
Jacopo Parretti
Tesi di Laurea Triennale in **Bioinformatica** presso il **Dipartimento di Informatica**, Universitá degli Studi di Verona
