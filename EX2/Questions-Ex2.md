## Questions
- [x] Task 4 says to use the "latest JDK". However, the videos recommend JDK 11 (specifically, Amazon Corretto 11). What should we go for?
    - We use Amazon Coretto 11.0.21 - Is that okay?
    - **Answer:** Yes, if it works! "Latest JDK" is just a recommendation. Use whichever works.
- [x] OpenCL: Is it required? We all get errors and warnings due to lacking OpenCL support.
  - **Answer:** Yes. OpenCL can be ignored. Affects mostly the density generation. 
- [x] 40-m run for Scenario 1: Start where? Pedestrian circle's center or the rightmost edge? Results will vary slightly!
  - **Answer:** Center (pedestrian's position)
- [x] Simulation is nondeterministic. Is it possible to standardize?
    - We get slightly different results on different machines using the same scenario file
        - Even using the same seeds!
    - **Answer:** We must set seeds to be used! There's a flag somewhere
- [x] Is it normal that the SIR model does not terminate by itself and we have to press the stop simulation button?
  - It also does not save the specified pedestrian groups in the output files, but SIR information are written correctly
  - **Answer:** It's part of the Optimal Steps button. Also, under "Simulation" tab, there's a way to specify how long the simulation runs.
- [x] Do we need to specify how we used the IDE or how we built the project? What about instructions to add the code in Tasks 4,5?
  - **Answer:** No build instructions are needed in the report unless it's required to replicate our code changes in Tasks 4,5. The goal is to make it so that a third party can replicate our results.
- [x] Are we supposed to use the groupid property of pedestrians or is it done differently in the SIR model? This property does not seem to get saved.
  - **Answer:** The group IDs are not usually saved under standard settings. That's why we need an appropriate processor and the SIR model.

### Extra pedantic questions just to be sure:
- [x] We use the source version of Vadere for tasks 1-3. Is that okay?
  - Answer: Yes!
- [x] We use the GitHub Master branch for Vadere instead of GitLab. Is that okay?
  - Answer: Yes, if it works! But the main work seems to be done on GitLab.