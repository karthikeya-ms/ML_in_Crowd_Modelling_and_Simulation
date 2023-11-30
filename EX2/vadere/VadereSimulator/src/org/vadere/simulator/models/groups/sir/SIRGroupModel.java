package org.vadere.simulator.models.groups.sir;


import org.vadere.annotation.factories.models.ModelClass;
import org.vadere.simulator.models.Model;
import org.vadere.simulator.models.groups.AbstractGroupModel;
import org.vadere.simulator.models.groups.Group;
import org.vadere.simulator.models.groups.GroupSizeDeterminator;
import org.vadere.simulator.models.groups.cgm.CentroidGroup;
import org.vadere.simulator.models.potential.fields.IPotentialFieldTarget;
import org.vadere.simulator.projects.Domain;
import org.vadere.state.attributes.Attributes;
import org.vadere.simulator.models.groups.sir.SIRGroup;
import org.vadere.state.attributes.models.AttributesSIRG;
import org.vadere.state.attributes.scenario.AttributesAgent;
import org.vadere.state.scenario.Agent;
import org.vadere.state.scenario.DynamicElementContainer;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.state.scenario.Topography;
import org.vadere.util.geometry.LinkedCellsGrid;
import org.vadere.util.geometry.shapes.VPoint;

import java.util.*;

/**
 * Implementation of groups for a susceptible / infected / removed (SIR) model.
 */
@ModelClass
public class SIRGroupModel extends AbstractGroupModel<SIRGroup> {

	private Random random;
	private LinkedHashMap<Integer, SIRGroup> groupsById;
	private Map<Integer, LinkedList<SIRGroup>> sourceNextGroups;
	private AttributesSIRG attributesSIRG;
	private Topography topography;
	private IPotentialFieldTarget potentialFieldTarget;
	private int totalInfected = 0;
	private double prevSimTimeInSec = 0.0;
	private double timeStepRemainder = 0.0;

	public SIRGroupModel() {
		this.groupsById = new LinkedHashMap<>();
		this.sourceNextGroups = new HashMap<>();
	}

	@Override
	public void initialize(List<Attributes> attributesList, Domain domain,
	                       AttributesAgent attributesPedestrian, Random random) {
		this.attributesSIRG = Model.findAttributes(attributesList, AttributesSIRG.class);
		this.topography = domain.getTopography();
		this.random = random;
        this.totalInfected = 0;
	}

	@Override
	public void setPotentialFieldTarget(IPotentialFieldTarget potentialFieldTarget) {
		this.potentialFieldTarget = potentialFieldTarget;
		// update all existing groups
		for (SIRGroup group : groupsById.values()) {
			group.setPotentialFieldTarget(potentialFieldTarget);
		}
	}

	@Override
	public IPotentialFieldTarget getPotentialFieldTarget() {
		return potentialFieldTarget;
	}

	private int getFreeGroupId() {
		if(this.random.nextDouble() < this.attributesSIRG.getInfectionRate()
        || this.totalInfected < this.attributesSIRG.getInfectionsAtStart()) {
			if(!getGroupsById().containsKey(SIRType.ID_INFECTED.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_INFECTED.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_INFECTED.ordinal(), g);
			}
            this.totalInfected += 1;
			return SIRType.ID_INFECTED.ordinal();
		}
		else{
			if(!getGroupsById().containsKey(SIRType.ID_SUSCEPTIBLE.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_SUSCEPTIBLE.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_SUSCEPTIBLE.ordinal(), g);
			}
			return SIRType.ID_SUSCEPTIBLE.ordinal();
		}
	}


	@Override
	public void registerGroupSizeDeterminator(int sourceId, GroupSizeDeterminator gsD) {
		sourceNextGroups.put(sourceId, new LinkedList<>());
	}

	@Override
	public int nextGroupForSource(int sourceId) {
		return Integer.MAX_VALUE/2;
	}

	@Override
	public SIRGroup getGroup(final Pedestrian pedestrian) {
		SIRGroup group = groupsById.get(pedestrian.getGroupIds().getFirst());
		assert group != null : "No group found for pedestrian";
		return group;
	}

	@Override
	protected void registerMember(final Pedestrian ped, final SIRGroup group) {
		groupsById.putIfAbsent(ped.getGroupIds().getFirst(), group);
	}

	@Override
	public Map<Integer, SIRGroup> getGroupsById() {
		return groupsById;
	}

	@Override
	protected SIRGroup getNewGroup(final int size) {
		return getNewGroup(getFreeGroupId(), size);
	}

	@Override
	protected SIRGroup getNewGroup(final int id, final int size) {
		if(groupsById.containsKey(id))
		{
			return groupsById.get(id);
		}
		else
		{
			return new SIRGroup(id, this);
		}
	}

	private void initializeGroupsOfInitialPedestrians() {
		// get all pedestrians already in topography
		DynamicElementContainer<Pedestrian> c = topography.getPedestrianDynamicElements();

		if (!c.getElements().isEmpty()) {
			// Here you can fill in code to assign pedestrians in the scenario at the beginning (i.e., not created by a source)
            //  to INFECTED or SUSCEPTIBLE groups. This is not required in the exercise though.

			for(Pedestrian p : c.getElements()) {
				if(!p.getGroupIds().isEmpty()) {
					assignToGroup(p, p.getGroupIds().getFirst());
				}
			}
		}
	}

	protected void assignToGroup(Pedestrian ped, int groupId) {
		SIRGroup currentGroup = getNewGroup(groupId, Integer.MAX_VALUE/2);
		currentGroup.addMember(ped);
		ped.getGroupIds().clear();
		ped.getGroupSizes().clear();
		ped.addGroupId(currentGroup.getID(), currentGroup.getSize());
		registerMember(ped, currentGroup);
	}

	protected void assignToGroup(Pedestrian ped) {
		int groupId = getFreeGroupId();
		assignToGroup(ped, groupId);
	}

	private boolean attemptRecover(Pedestrian infected) {
		if (this.random.nextDouble() < this.attributesSIRG.getRecoveryRate()) {
			elementRemoved(infected);
			assignToGroup(infected, SIRType.ID_REMOVED.ordinal());
			return true;
		}
		return false;
	}


	/* DynamicElement Listeners */

	@Override
	public void elementAdded(Pedestrian pedestrian) {
		assignToGroup(pedestrian);
	}

	@Override
	public void elementRemoved(Pedestrian pedestrian) {
		Group group = groupsById.get(pedestrian.getGroupIds().getFirst());
		if (group.removeMember(pedestrian)) { // if true pedestrian was last member.
			groupsById.remove(group.getID());
		}
	}

	/* Model Interface */

	@Override
	public void preLoop(final double simTimeInSec) {
		initializeGroupsOfInitialPedestrians();
		topography.addElementAddedListener(Pedestrian.class, this);
		topography.addElementRemovedListener(Pedestrian.class, this);
	}

	@Override
	public void postLoop(final double simTimeInSec) {
	}

	@Override
	public void update(final double simTimeInSec) {
		// check the positions of all pedestrians and switch groups to INFECTED (or REMOVED).
		final DynamicElementContainer<Pedestrian> c = topography.getPedestrianDynamicElements();

		//independent time step. Calculates:
		//1- the step resolution
		//2- adds the resolution to timeStepRemainder
		//3- reassigns prevSimTimeInSec
		double deltaTime = simTimeInSec - this.prevSimTimeInSec;
		this.timeStepRemainder += deltaTime;
		this.prevSimTimeInSec = simTimeInSec;

		//4- update only if we have reached the time resolution
		if (this.timeStepRemainder < attributesSIRG.getSIRTimeResolution()) {
			return;
		}

		//5- keep the step remainder
		this.timeStepRemainder %= attributesSIRG.getSIRTimeResolution();

		// The following is a more efficient method to get the neighbors of infected pedestrians
		final LinkedCellsGrid<Pedestrian> linkedCellsGrid = topography.getSpatialMap(Pedestrian.class);

		/* Just like the original implementation, we loop over the infected pedestrians first, and get the pedestrians within its radius.
		This means that if a susceptible pedestrian is near N infected pedestrians, their infection rate is 4 times higher */
		c.getElements()
				.stream()
				// Get only infected
				.filter(ped -> getGroup(ped).getID() == SIRType.ID_INFECTED.ordinal())
				// Attempt to recover infected first if recoverBeforeSpread is true
				.filter(infected -> !attributesSIRG.getRecoverBeforeSpread() || !attemptRecover(infected))
				.forEach(infected -> {
						// No need to calculate dist for all other pedestrians anymore compared to original implementation. O(1) access to supplied radius
						final Collection<Pedestrian> neighbors = linkedCellsGrid.getObjects(infected.getPosition(), attributesSIRG.getInfectionMaxDistance());
						// O(N_neighbors) rather than O(N_pedestrians)
						neighbors.stream()
								// We only care about the susceptible pedestrians
								// Note that this also filters out the infected pedestrian whose position is `infectedPos`
								.filter(p -> getGroup(p).getID() == SIRType.ID_SUSCEPTIBLE.ordinal())
								// We randomly infect our susceptible pedestrians
								.forEach(susceptiblePed -> {
									if (this.random.nextDouble() < attributesSIRG.getInfectionRate()) {
										elementRemoved(susceptiblePed);
										assignToGroup(susceptiblePed, SIRType.ID_INFECTED.ordinal());
									}
								});

						if (!attributesSIRG.getRecoverBeforeSpread()) {
							attemptRecover(infected);
						}
					}
				);
	}
}
