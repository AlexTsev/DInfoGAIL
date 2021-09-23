!!![Οι κώδικες έχουν ελεγχθεί μόνο σε Windows 10]
Τα projects για το Hopper και το Αviation είναι ξεχωριστά.

ΒΗΜΑ1:
σε python version 3.6 με anaconda:
https://www.anaconda.com/products/individual 
στον editor PyCharm:
https://www.jetbrains.com/pycharm/

ΒΗΜΑ2:
ίσως χρειάζονται κάποιες εκδόσεις των βιβλιοθηκών του scikit-learn==0.23.2, matplotlib==3.3.2, pandas==1.1.2, numpy==1.19.1 και seaborn==0.11.0
*(Οποιαδήποτε βιβλιοθήκη έχει θέμα να την βάλετε από το terminal μπορείτε να την βάλετε manually με το αντίστοιχο version της μεσα από τα settings των βιβλιοθηκών του project στο Pycharm)

ΒΗΜΑ3:
*(Για την χρησιμοποιήση της gpu κυρίως στον 2ο αλγόριθμο χρειαζεται τα plugins της nvidia cuda και cudnn, υπάρχουν manuals για τα versions αναλόγως την gpu)
*(τις εντολες 'pip install..' τις πατάμε στο terminal του Pycharm).

Για το pre-training του VAE χρειάζεται μόνο η βιβλιοθήκη του
tensorflow == 1.14.0('pip install tensorflow == 1.14.0') 
χωρις GPU για να γίνει πιο γρήγορο training

Για το 2ο κομμάτι με τον GAIL χρειάζονται και οι δύο βιβλιοθήκες
tensorflow == 1.14.0('pip install tensorflow == 1.14.0')
tensorflow-gpu == 1.14.0('pip install tensorflow-gpu == 1.14.0')

ΒΗΜΑ4(ΓΙΑ HOPPER):
Για να τρέξει το περιβάλλον του Hopper είναι αναγκαίο να υπάρχει license στα περιβάλλοντα του mujoco-OpenAI-gym που περνάμε το key μέσα στον editor συγκεκριμένα στο PyCharm:
https://www.roboti.us/license.html
http://www.mujoco.org/
Το δικό μου license είναι μόνο για ένα pc οπότε δεν μπορώ να το διαθέσω, μπορείτε να πάρετε 1 στην σελίδα όμως.
Επίσης χρειάζεται η εγκατάσταση του φακέλου mjpro131 από την ιστοσελίδα https://www.roboti.us/index.html στο tab Products mjpro131 win64 και το βάζετε το path στο Path στα system environment variables στα Windows.
Μπορείτε να βρείτε τα environment variables στο search των windows.

Το περιβάλλον του aviation δεν χρειάζεται κάποιο license ή κάποια εγκατάσταση.

ΒΗΜΑ5(ΓΙΑ HOPPER):
Αφου έχει περαστεί το key ειναι απαραίτητο να έχει περαστεί στο PYCHARM Editor για το Hopper η βιβλιοθήκη του mujoco-py και του OpenAI-gym συγκεκριμένα τα versions:
mujoco-py==0.5.7
gym==0.17.3
----------------------------------------------
ΒΗΜΑ6(ΓΙΑ HOPPER):
Επειδή υπήρχαν κάποια θέματα στην εγκατάσταση αφού περάσετε τις βιβλιοθήκες με τις εντολές 'pip install mujoco-py==0.5.7' & 'pip install gym==0.17.3' για το Hopper
έχω κάνει κάποιες μετατροπές καλός ή κακός στους κώδικες των βιβλιοθηκών που θα χρειαστούν με αυτά τα versions γιατί μετά αλλάζουν οι εντολές στον κώδικα στα καινούργια versions!!
αυτά τα αρχεία που πρέπει να τα κάνετε replace βρίσκονται μέσα στον φάκλεο Diplwmatikh/libraries for mujoco edited/...!!!!!!!!!!!!!!!!! και είναι κυρίως για το rendering στο testing.
Αν τα κάνετε replace δεν θα χρειαστεί να κάνετε κάποια αλλαγή εκτός αν θέλετε να δείτε σε οθόνη τα modes στο testing του Hopper που το εξηγώ μετά.

!!! ΔΕΝ ΘΑ ΧΡΕΙΑΣΤΕΙ ΑΝ ΚΑΝΕΤΕ REPLACE τα αρχεια
Είχα παρατηρήσει ότι δημιουργούσε ένα error επειδή δεν μπορούσε να ανιχνεύσει την πλατφόρμα ότι είναι σε windows και χρειάστηκε να αλλάξω  στο ίδιο path με πριν στο σημείο που γίνονται εγκατάσταση τα libraries της python στο hopper project
.\anaconda3.1\envs\HopperEnv_Directed_Info_GAIL\Lib\site-packages\mujoco_py\mjlib.py αυτή την σειρα στην 13-14η γραμμή, να έχει κατάληξη .dll

elif sys.platform.startswith("win"):
    libfile = os.path.join(path_prefix, "bin/mujoco131.dll")

και πάλι στην ίδια τοποθεσία .\anaconda3.1\envs\HopperEnv_Directed_Info_GAIL\Lib\site-packages\mujoco_py\platname_targdir.py
μετά τα if statements βάζετε στην 8η-9η σειρα.

platname = "win"
targdir = "mujoco_%s"%platname

!!!

ΒΗΜΑ7:
Εφόσον θέλετε να τρέξετε το hopper χωρίς να εμφανίζονται τα modes σε οθόνη rendering της openGL το αφήνετε όπως έχει μετα τις μετατροπές αλλιώς αν θέλετε να φαίνονται πρέπει να αλλάξετε αυτό το αρχείο στην τοποθεσια του enviroment που εχει τα libraries, σε εμενα ηταν στο path:
.\anaconda3.1\envs\HopperEnv_Directed_Info_GAIL\Lib\site-packages\mujoco_py\mjviewer.py , αυτό το αρχείο κάνετε replace με αυτό που έχει δωθεί είναι mjviewer-edited.py και το μετεονομάζεται σε mjviewer.py.
------------------------------------------------
ΒΗΜΑ8:
Επειδή κάθε υλοποίηση έχει διαφορετικά features και αλλάζουν και τα modes, όλα τα αρχεία της κάθε υλοποίησης βρίσκονται σε φακέλους μέσα στον φάκελο ./Dir_Info_GAIL/.. και πρέπει να τα κάνετε copy τα αρχεία 
(για hopper: main.py, trpo.py ή για aviation: preprocess.py,trpo.py) στον main folder αναλόγως ποια υλοποίηση θέλετε να τρέξετε, 
καλύτερα να γίνει manually και όχι μέσα απο κάποιον editor ο οποίος μπορεί να αλλάξει τα paths μέσα στον κώδικα.

στο φακελο ./VAE/results_gumbel_softmax/checkpoint/.. έχει ολα τα checkpoints από τα πειράματα στο pre-training που χρησιμοποιούνται με τα paths και στον GAIL
Στη κάθε διαφορετική υλοποίηση μέσα στον φάκελο αυτό έχει τα plots απτα training losses και τα training και testing latent variables και του φακελους με τα checkpoints.

στον φακελο ./runs/... έχει ολα τα αποτελέσματα των πειραμάτων για το behavioral cloning-GAIL και Directed-Info GAIL

το αρχείο στον φάκελο του project του hopper ./experiments_means.py περιέχει ολα τα αποτελέσματα των 5 πειραμάτων από τον Directed-Info GAIL από τα 2 περιβάλλοντα και υπολογίζει το mean τους

το αρχείο ./create_hopper_dataset.py δημιουργεί το expert dataset του hopper εγώ το χω φτιάξει ήδη μέσα στον φάκελο ./dataset/...

ΒΗΜΑ9:
!!!
-στο hopper 
το βασικό αρχείο για να τρέξει το pre-training βρίσκεται στον φάκελο 
./VAE/vae_gumbel_softmax_edited_last - batch(3modes).py ή ./VAE/vae_gumbel_softmax_edited_last - batch(5modes).py
ενώ για να τρέξουμε τον 2ο αλγόριθμο τρέχουμε το ./main.py αναλόγως ποια modes θέλουμε να τρεξουμε τα βρίσκουμε στον φάκελο ./Dir_Info_GAIL/3modes/.. ή ./Dir_Info_GAIL/5modes/..
στον φακελο with_openGL_mode περιέχονται τα αντίστοιχα αρχεία αλλά με την εντολή για να εμφανίζονται και τα modes. 
Προσοχή!! όταν αλλάξουμε το mjviewer.py αρχείο με το rendering των modes θέλουμε το main.py, trpo.py που βρίσκετε στο φάκελο with_openGL_mode αλλιώς αν δεν θελουμε rendering των modes κρατάμε το αρχικό mjviewer.py με 
τα αντίστοιχα main.py,trpo.py που βρισκονται στον φάκελο ./Dir_Info_GAIL/3modes/main.py ή ./Dir_Info_GAIL/5modes/main.py

-στο aviation 
το βασικό αρχείο για να τρέξει το pre-training βρίσκεται στον φάκελο 
features με raw trajectories: ./VAE/vae_gumbel_softmax_edited_last - batch - RAW.py
features με NOAA: ./VAE/vae_gumbel_softmax_edited_last - batch.py
features με NOAA+METAR: ./VAE/vae_gumbel_softmax_edited_last - batch - METAR.py
ενώ για να τρέξουμε τον 2ο αλγόριθμο τρέχουμε το ./preprocess.py 
τα αρχεία preprocess.py, trpo.py υπαρχουν στους φακέλους ./Dir_Info_GAIL/ΝΟΑ/3modes ή 5modes , ./Dir_Info_GAIL/METAR/3modes ή 5modes, ./Dir_Info_GAIL/RAW/3modes ή 5modes
!!

Τα αρχεία ./VAE/renameVAE.py και  ./VAE/rename.py χρησιμοποιούνται για να μεταονομάσουν τα weights, biases και layers από τα checkpoints του pre-training που θα χρησιμοποιηθούν στο testing του VAE είτε στο training/testing του Directed-Info GAIL αντίστοιχα.
σε αυτά ορίζουμε τα αντίστοιχα paths για να φτιαχτούν τα αρχεία.Αυτά τα paths καλούνται και στο run του Directed-Info GAIL στα αρχεία trpo.py που τα χω ορίσει ήδη για τα πειράματα που έχω κάνει.

Σε Καθε experiment που θα γινει στο pre-training πρέπει να έχουμε τους φακέλους ακριβώς όπως είναι για παράδειγμα στο ./VAE/results_gumbel_softmax/checkpoint/run9(5modes) 
οπότε οταν ονομάσουμε το experiment με το αντιστοιχο νούμερο στο αρχείο του pre-training στην αρχη του πρεπει να φτιάξουμε τον αντίστοιχο φάκελο με τους κενούς φακελους μεσα οπως ειναι στο πάνω παραδειγμα run9(5modes).
Ωστε τα αρχεια που δημιουργουνται να πανε εκει.

Για το aviation που δεν εχει rendering τα αρχεία-αποτελέσματα csv που δημιουργούνται στον main folder από τον Directed-Info GAIL μπορείτε να τα δείτε στα maps στο QGIS Desktop version 3.14.16 πρόγραμμα



