# arXiv_classification

Si sviluppa un algoritmo che classifichi in base al loro argomento gli articoli scientifici contenuti in
una frazione del dataset di arXiv.


1. Classificazione supervisionata:

	  • usando come labels le categorie associate a un articolo e come feature il testo
	  risultante dall’unione di abstract e titolo;
	
	  • usando come labels le categorie associate agli articoli e come feature una lista di
	  parole chiave estratta dal testo con un transformer;
	  
	  • usando come label una parola chiave e come feature il testo (restringendosi a una
	  sola macrocategoria).


3. Non supervisionata:
• usando solamente l’abstract e il titolo dell'articolo.
