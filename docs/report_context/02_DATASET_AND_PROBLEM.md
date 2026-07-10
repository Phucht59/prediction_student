# Dataset and problem definition

Source: UCI Student Performance, `student-mat`; 395 observations from Portuguese
secondary-school mathematics. It is not a Vietnamese university dataset. The
raw schema has 33 columns: school, sex, age, address, famsize, Pstatus, Medu,
Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup,
famsup, paid, activities, nursery, higher, internet, romantic, famrel,
freetime, goout, Dalc, Walc, health, absences, G1, G2 and G3.

G3 is excluded from model features. Class definition: Low G3<=9 (130), Medium
10<=G3<=14 (192), High G3>=15 (73). The locked-test support is Low 26, Medium
38, High 15. The fixed stratified split is 316 development / 79 locked test.

G2 is expected to be powerful because it is a prior course grade near final
assessment. This is why G2-only rules are mandatory baselines. Late-stage uses
G1/G2; early-warning excludes G2; pre-assessment excludes both. Generalization
is limited by one subject, one historical population and 395 records.
