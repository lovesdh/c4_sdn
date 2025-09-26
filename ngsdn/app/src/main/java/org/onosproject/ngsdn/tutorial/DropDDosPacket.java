package org.onosproject.ngsdn.tutorial;



import org.onlab.packet.Ip6Address;
import org.onlab.packet.Ip6Prefix;
import org.onlab.packet.IpAddress;
import org.onlab.packet.IpPrefix;

import org.onlab.util.ItemNotFoundException;

import org.onosproject.core.ApplicationId;
import org.onosproject.mastership.MastershipService;

import org.onosproject.net.DeviceId;

import org.onosproject.net.config.NetworkConfigService;
import org.onosproject.net.device.DeviceEvent;
import org.onosproject.net.device.DeviceListener;
import org.onosproject.net.device.DeviceService;
import org.onosproject.net.flow.FlowRule;
import org.onosproject.net.flow.FlowRuleOperations;
import org.onosproject.net.flow.FlowRuleService;
import org.onosproject.net.flow.criteria.PiCriterion;
import org.onosproject.net.group.GroupDescription;
import org.onosproject.net.group.GroupService;
import org.onosproject.net.host.HostService;

import org.onosproject.net.pi.model.PiActionId;
import org.onosproject.net.pi.model.PiMatchFieldId;
import org.onosproject.net.pi.runtime.PiAction;
import org.onosproject.ngsdn.tutorial.common.FabricDeviceConfig;
import org.onosproject.ngsdn.tutorial.common.Utils;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import static org.onosproject.ngsdn.tutorial.AppConstants.INITIAL_SETUP_DELAY;

@Component(
    immediate = true,
    enabled = true
)
public class DropDDosPacket {
    private static final Logger log =
            LoggerFactory.getLogger(NdpReplyComponent.class.getName());
    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected FlowRuleService flowRuleService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected DeviceService deviceService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected GroupService groupService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected MastershipService mastershipService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected MainComponent mainComponent;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected HostService hostService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected NetworkConfigService configService;

    private DeviceListener deviceListener = new InternalDeviceListener();

    private ApplicationId appId;

    final int DEFAULT_DROP_GROUP = 0XEC3C0000;
    private static final long GROUP_INSERT_DELAY_MILLIS = 200;

    // final Collection<String> DDosIPSet = {
    //     ""
    // }
    private final static Ip6Address DROP_IP6_ADDRESS = Ip6Address.valueOf("2001:1:1::a");
    
    @Activate
    public void activate(){
        appId  = mainComponent.getAppId();

        deviceService.addListener(deviceListener);

        mainComponent.scheduleTask(this::setUpAllDevices, INITIAL_SETUP_DELAY);
        log.info ("Started");
        System.out.println("fucker");
    }
    @Deactivate
    public void deactivate(){
        deviceService.removeListener(deviceListener);
        log.info("Stopped");
    }



    private GroupDescription createDDosGroup(int groupId,DeviceId deviceId){
        final String tableId = "IngressPipeImpl.ddos_drop_table";

        final PiAction action = PiAction.builder()
                                .withId(PiActionId.of("NoAction"))
                                .build();
        final List<PiAction> actions =  Lists.newArrayList();
        actions.add(action);
        return Utils.buildDDosDropGroup(deviceId, tableId, groupId,actions, appId);
    }              
    
    

    private FlowRule createDropFlowRules(int groupId,
                                    Ip6Prefix ip6Prefix,DeviceId deviceId){
        // log.info("Adding Drop flow Rule on {}...",deviceId);
        final String tableId = "IngressPipeImpl.ddos_drop_table";
        final PiCriterion match = PiCriterion.builder()
                .matchLpm(
                    PiMatchFieldId.of("hdr.ipv6.src_addr"),
                    ip6Prefix.address().toOctets(),
                    ip6Prefix.prefixLength())
                .build();
        final PiAction action = PiAction.builder()
                                .withId(PiActionId.of("NoAction"))
                                 .build();
        final FlowRule rule = Utils.buildFlowRule(deviceId, appId, tableId, match, action);
        return rule;
        // final PiTableAction action = PiActionProfileGroupId.of(groupId);
        // return Utils.buildFlowRule(deviceId, appId,tableId, match, action);
    }


    public class InternalDeviceListener implements DeviceListener{
        @Override 
        public boolean isRelevant(DeviceEvent event){
            switch(event.type()){
                case DEVICE_ADDED:
                case DEVICE_AVAILABILITY_CHANGED:
                    break;
                default:
                    return false;
            }
            final DeviceId deviceId = event.subject().id();
            return mastershipService.isLocalMaster(deviceId);
        }

        @Override 
        public void event(DeviceEvent event){
            mainComponent.getExecutorService().execute(() -> {
                DeviceId deviceId = event.subject().id();
                log.info("{} event! device id {}",event.type(),deviceId);
                // final Host DDosHost = hostService.getHostsByIp();
                setUpDevice(deviceId);
            });
        }
    }

    private void setDDosDropRules(DeviceId deviceId,Collection<Ip6Address> ip6Addresses){

        final GroupDescription group = createDDosGroup(DEFAULT_DROP_GROUP, deviceId);
        final List<FlowRule> flowRules = ip6Addresses.stream()
                            .map(IpAddress::toIpPrefix)
                            .filter(IpPrefix::isIp6)
                            .map(IpPrefix::getIp6Prefix)
                            .map(prefix -> createDropFlowRules(DEFAULT_DROP_GROUP,prefix,deviceId))
                            .collect(Collectors.toList());
        System.out.println("fucker11");
        insertInOrder(group, flowRules);
    }


    
    private void insertInOrder(GroupDescription group, Collection<FlowRule> flowRules) {
        try {
            groupService.addGroup(group);
            // Wait for groups to be inserted.
            Thread.sleep(GROUP_INSERT_DELAY_MILLIS);
            flowRules.forEach(flowRuleService::applyFlowRules);
        } catch (InterruptedException e) {
            log.error("Interrupted!", e);
            Thread.currentThread().interrupt();
        }
    }
    // private synchronized void setUpAllDevices(){
    //     stream(deviceService.getAvailableDevices())
    //         .map(Device::id)
    //         .filter(mastershipService::isLocalMaster)
    //         .forEach(deviceId -> {
    //             log.info("DDOs Drop Starting initial");
    //             List<Ip6Address> ip6Addresses = Lists.newArrayList();
    //             ip6Addresses.add(DROP_IP6_ADDRESS);
    //             setDDosDropRules(deviceId, ip6Addresses);
    //         });
                
    // }
    private void setUpAllDevices() {
        deviceService.getAvailableDevices().forEach(device -> {
            if (mastershipService.isLocalMaster(device.id())) {
                log.info("*** NDP REPLY - Starting Initial set up for {}...", device.id());
                setUpDevice(device.id());
            }
        });
    }
    private void setUpDevice(DeviceId deviceId){
        final FabricDeviceConfig config = configService.getConfig(
            deviceId, FabricDeviceConfig.class);
        if (config == null) {
            // Config not available yet
            throw new ItemNotFoundException("Missing fabricDeviceConfig for " + deviceId);
        }
        // final List<PiAction> actions =  Lists.newArrayList();
        // actions.add(action);
        final List<Ip6Address> ip6Addresses = Lists.newArrayList();
        ip6Addresses.add(DROP_IP6_ADDRESS);

        final List<FlowRule> flowRules = ip6Addresses.stream()
                                        .map(IpAddress::toIpPrefix)
                                        .filter(IpPrefix::isIp6)
                                        .map(IpPrefix::getIp6Prefix)
                                        .map(perfix -> createDropFlowRules(DEFAULT_DROP_GROUP, perfix, deviceId))
                                        .collect(Collectors.toList());

        System.out.println("device:" + deviceId +"flows:" + flowRules.size());

        installRules(flowRules);
    }

    private void installRules(Collection<FlowRule> flowRules) {
        FlowRuleOperations.Builder ops = FlowRuleOperations.builder();
        flowRules.forEach(ops::add);
        flowRuleService.apply(ops.build());
    }
}
