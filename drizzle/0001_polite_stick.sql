CREATE TABLE `codeCommands` (
	`id` int AUTO_INCREMENT NOT NULL,
	`command` text NOT NULL,
	`output` text,
	`error` text,
	`status` enum('pending','running','completed','failed') NOT NULL DEFAULT 'pending',
	`duration` int,
	`userId` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `codeCommands_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `events` (
	`id` int AUTO_INCREMENT NOT NULL,
	`type` varchar(64) NOT NULL,
	`category` varchar(64) NOT NULL,
	`title` text NOT NULL,
	`data` text NOT NULL,
	`metadata` text,
	`status` enum('pending','running','completed','failed') NOT NULL DEFAULT 'pending',
	`userId` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `events_id` PRIMARY KEY(`id`)
);
