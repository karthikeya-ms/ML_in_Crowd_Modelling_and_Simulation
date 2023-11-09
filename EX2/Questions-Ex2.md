## Questions
- Task 4 says to use the "latest JDK". However, the videos recommend JDK 11 (specifically, Amazon Corretto 11). What should we go for?
    - We use Amazon Coretto 11.0.21 - Is that okay? 
- OpenCL: Is it required? We all get errors and warnings due to lacking OpenCL support.
- 40-m run for Scenario 1: Start where? Pedestrian circle's center or the rightmost edge? Results will vary slightly!
- Simulation is nondeterministic. Is it possible to standardize?
    - We get slightly different results on different machines using the same scenario file
        - Even using the same seeds!
- Is it normal that the SIR model does not terminate by itself and we have to press the stop simulation button?
  - It also does not save the specified pedestrian groups in the output files, but SIR information are written correctly 

### Extra pedantic questions just to be sure:
- We use the source version of Vadere for tasks 1-3. Is that okay?
- We use the GitHub Master branch for Vadere instead of GitLab. Is that okay?