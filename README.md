# Σχεδιασμός και Αξιολόγηση Συστημάτων Ομοσπονδιακής Μάθησης για την Παραγωγή Προτάσεων με βάση γράφους

### Design and Evaluation of Federated Learning Systems for Graph-Based Recommendation

ΔΙΠΛΩΜΑΤΙΚΗ ΕΡΓΑΣΙΑ --- ΠΟΛΙΤΗΣ ΕΥΣΤΡΑΤΙΟΣ

DIPLOMA THESIS --- POLITIS EFSTRATIOS

## Πληροφορίες (GR)
Το παρόν repository αφορά την διπλωματική μου εργασία στο [Τμήμα Ηλεκτρολόγων Μηχανικών και Μηχανικών Η/Υ](https://www.ee.duth.gr/) του Δημοκριτείου Πανεπιστημίου Θράκης, η οποία ολοκληρώθηκε τον Αύγουστο του 2026 υπό την επίβλεψη του Καθηγητή κ. Παύλου Εφραιμίδη.

Η εργασία πραγματεύεται την χρήση ενός [Graph Neural Network](https://en.wikipedia.org/wiki/Graph_neural_network) για την υλοποίηση ενός [Συστήματος Σύστασης](https://en.wikipedia.org/wiki/Recommender_system) για [IoT συσκευές](https://en.wikipedia.org/wiki/Internet_of_things) στα πλαίσια ενός [Federated Learning](https://en.wikipedia.org/wiki/Federated_learning) περιβάλλοντος.

## Information (EN)
This repository contains my diploma thesis for the [Department of Electrical and Computer Engineering](https://www.ee.duth.gr/en/) at Democritus University of Thrace, which was completed in August 2026 under the supervision of Professor Pavlos Efremidis.

The thesis addresses the use of a [Graph Neural Network](https://en.wikipedia.org/wiki/Graph_neural_network) to implement a [Recommendation System](https://en.wikipedia.org/wiki/Recommender_system) for [IoT devices](https://en.wikipedia.org/wiki/Internet_of_things) within a [Federated Learning](https://en.wikipedia.org/wiki/Federated_learning) environment.

## Περίληψη (GR)
Η ραγδαία εξάπλωση των συσκευών του Διαδικτύου των Πραγμάτων (IoT) έχει καταστήσει τη σύσταση αυτοματισμών (κανόνων της μορφής «αν συμβεί το Α, τότε εκτέλεσε το Β») ένα ολοένα και πιο σημαντικό πρόβλημα. Τα δεδομένα που απαιτούνται για την εκπαίδευση ενός τέτοιου συστήματος είναι όμως εξ ορισμού ευαίσθητα, γεγονός που καθιστά την κεντρικοποιημένη συλλογή τους προβληματική. Η παρούσα εργασία μελετά το πρόβλημα της σύστασης κανόνων IoT σε ομοσπονδιακό πλαίσιο, ορμώμενη από τον ανοικτό διαγωνισμό της εταιρείας Wyze και τη λύση της δεύτερης θέσης.

Το πρόβλημα επαναδιατυπώνεται ως ομοσπονδιακή πρόβλεψη συνδέσεων σε γράφους ανά χρήστη και αναπτύσσονται τρεις διαδοχικές εκδόσεις μοντέλου αυξανόμενης πολυπλοκότητας, καταλήγοντας σε ένα relational μοντέλο που συνδυάζει τον CompGCN encoder, τον ComplEx decoder και τη συνάρτηση σφάλματος InfoNCE. Το μοντέλο αυτό βελτιώνει την κεντρικοποιημένη λύση αναφοράς κατά 71% στη μετρική MRR. Η συστηματική αξιολόγηση σε τέσσερα σενάρια ενορχήστρωσης (κεντρικοποιημένο, cross-silo και δύο παραλλαγές cross-device) αποκαλύπτει ότι η βέλτιστη επιλογή μοντέλου εξαρτάται από τον τύπο της ενορχήστρωσης: το ισχυρότερο μοντέλο κυριαρχεί παντού, εκτός από το πιο κατανεμημένο σενάριο, όπου μια απλούστερη έκδοση αποδεικνύεται ανθεκτικότερη.

## Abstract (EN)
The rapid proliferation of Internet of Things (IoT) devices has made the recommendation of automations (rules of the form "if A happens, then perform B") an increasingly important problem. The data required to train such a system is, however, inherently sensitive, which makes its centralized collection problematic. This thesis studies the problem of IoT rule recommendation in a federated setting, taking as its starting point Wyze's open competition and the second-place solution as a reference baseline.

The problem is reformulated as federated link prediction over per-user graphs, and three successive model versions of increasing complexity are developed, culminating in a relational model that combines a CompGCN encoder, a ComplEx decoder, and an InfoNCE loss. This model improves upon the centralized reference solution by 71% in Mean Reciprocal Rank (MRR). A systematic evaluation across four orchestration scenarios (centralized, cross-silo, and two cross-device variants) reveals that the optimal model choice depends on the type of orchestration: the strongest model dominates everywhere except the most distributed scenario, where a simpler version proves more robust.

## Περιεχόμενα
Ο φάκελος `code/` περιέχει τον πλήρη κώδικα της εργασίας, ενώ ο φάκελος `thesis/` περιέχει τα απαραίτητα αρχεία .tex για την παραγωγή του τελικού παραδοτέου (το οποίο βρίσκεται και αυτούσιο στο `thesis/thesis.pdf`).

## Contents
The `code/` folder contains the complete source code for the project, while the `thesis/` folder contains the necessary .tex files for generating the final deliverable (which is also available as-is in `thesis/thesis.pdf`).